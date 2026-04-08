#![allow(non_camel_case_types)]

use std::ffi::c_char;
use std::ffi::c_int;
use std::ffi::c_void;
use std::sync::Arc;

use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::expr::stats::Precision::Exact;
use vortex::expr::stats::Precision::Inexact;
use vortex::file::multi::MultiFileDataSource;
use vortex::io::runtime::BlockingRuntime;
use vortex::scan::DataSource;
use vortex::scan::DataSourceRef;

use crate::RUNTIME;
use crate::dtype::vx_dtype;
use crate::error::try_or_default;
use crate::error::vx_error;
use crate::session::vx_session;
use crate::to_string;

crate::arc_dyn_wrapper!(
    /// A reference to one or more (possibly remote) files.
    /// Creating vx_data_source opens the first matched file to read the schema.
    /// All other I/O is deferred until a scan is requested. Multiple scans may
    /// be requested from a single data source.
    dyn DataSource,
    vx_data_source);

pub struct VxFileHandle;
pub type vx_file_handle = *const VxFileHandle;

pub type vx_list_callback =
    Option<unsafe extern "C" fn(userdata: *mut c_void, name: *const c_char, is_dir: c_int)>;
pub type vx_glob_callback =
    Option<unsafe extern "C" fn(userdata: *mut c_void, file: *const c_char)>;

pub type vx_fs_set_userdata = Option<unsafe extern "C" fn(userdata: *mut c_void)>;

pub type vx_fs_open = Option<
    unsafe extern "C" fn(userdata: *mut c_void, path: *const c_char, err: *mut *mut vx_error),
>;
pub type vx_fs_create = Option<
    unsafe extern "C" fn(userdata: *mut c_void, path: *const c_char, err: *mut *mut vx_error),
>;

pub type vx_fs_list = Option<
    unsafe extern "C" fn(
        userdata: *const c_void,
        path: *const c_char,
        callback: vx_list_callback,
        error: *mut *mut vx_error,
    ),
>;

pub type vx_fs_close = Option<unsafe extern "C" fn(handle: vx_file_handle)>;
pub type vx_fs_size =
    Option<unsafe extern "C" fn(handle: vx_file_handle, err: *mut *mut vx_error) -> u64>;

pub type vx_fs_read = Option<
    unsafe extern "C" fn(
        handle: vx_file_handle,
        offset: u64,
        len: usize,
        buffer: *mut u8,
        err: *mut *mut vx_error,
    ) -> u64,
>;

pub type vx_fs_write = Option<
    unsafe extern "C" fn(
        handle: vx_file_handle,
        offset: u64,
        len: usize,
        buffer: *mut u8,
        err: *mut *mut vx_error,
    ) -> u64,
>;

pub type vx_fs_sync = Option<unsafe extern "C" fn(handle: vx_file_handle, err: *mut *mut vx_error)>;

pub type vx_glob = Option<
    unsafe extern "C" fn(glob: *const c_char, callback: vx_glob_callback, err: *mut *mut vx_error),
>;

/// Options for creating a data source.
///
/// "files" is the only required field.
/// If any "fs_*" callback is provided, all of them must be provided.
#[repr(C)]
#[cfg_attr(test, derive(Default))]
pub struct vx_data_source_options {
    /// Required: files names. May be a glob pattern like "*.vortex".
    pub files: *const c_char,
    pub fs_set_userdata: vx_fs_set_userdata,
    pub fs_open: vx_fs_open,
    pub fs_create: vx_fs_create,
    pub fs_list: vx_fs_list,
    pub fs_close: vx_fs_close,
    pub fs_size: vx_fs_size,
    pub fs_read: vx_fs_read,
    pub fs_write: vx_fs_write,
    pub fs_sync: vx_fs_sync,
    pub glob: vx_glob,
}

// TODO(myrrc): see https://github.com/vortex-data/vortex/issues/7324
#[cfg(vortex_nightly)]
unsafe extern "C" {
    pub fn __lsan_disable();
    pub fn __lsan_enable();
}

unsafe fn data_source_new(
    session: *const vx_session,
    opts: *const vx_data_source_options,
) -> VortexResult<*const vx_data_source> {
    vortex_ensure!(!session.is_null());
    vortex_ensure!(!opts.is_null());

    let session = vx_session::as_ref(session);

    let opts = unsafe { &*opts };
    vortex_ensure!(!opts.files.is_null());

    let glob = unsafe { to_string(opts.files) };

    #[cfg(vortex_nightly)]
    unsafe {
        __lsan_disable();
    }

    let ds = RUNTIME.block_on(async {
        let data_source = MultiFileDataSource::new(session.clone())
            //.with_filesystem(fs)
            .with_glob(glob)
            .build()
            .await?;
        Ok(vx_data_source::new(Arc::new(data_source) as DataSourceRef))
    });

    #[cfg(vortex_nightly)]
    unsafe {
        __lsan_enable();
    }

    ds
}

/// Create a data source.
/// The first matched file is opened eagerly. to read the schema. All other I/O
/// is deferred until a scan is requested. The returned pointer is owned by the
/// caller and must be freed with vx_data_source_free.
///
/// On error, sets "err" and returns NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_data_source_new(
    session: *const vx_session,
    opts: *const vx_data_source_options,
    err: *mut *mut vx_error,
) -> *const vx_data_source {
    try_or_default(err, || unsafe { data_source_new(session, opts) })
}

/// Return the schema of the data source as a non-owned dtype.
/// The returned pointer is valid as long as "ds" is alive. Do not free it.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_data_source_dtype(ds: *const vx_data_source) -> *const vx_dtype {
    vx_dtype::new_ref(vx_data_source::as_ref(ds).dtype())
}

#[repr(C)]
#[cfg_attr(test, derive(PartialEq, Debug))]
enum vx_cardinality {
    VX_CARD_UNKNOWN = 0,
    VX_CARD_ESTIMATE = 1,
    VX_CARD_MAXIMUM = 2,
}

#[repr(C)]
pub struct vx_data_source_row_count {
    cardinality: vx_cardinality,
    /// Set only when "cardinality" is not VX_CARD_UNKNOWN
    rows: u64,
}

/// Write data source's row count estimate into "row_count".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_data_source_get_row_count(
    ds: *const vx_data_source,
    row_count: *mut vx_data_source_row_count,
) {
    let rc = unsafe { &mut *row_count };
    match vx_data_source::as_ref(ds).row_count() {
        Some(Exact(rows)) => {
            rc.cardinality = vx_cardinality::VX_CARD_MAXIMUM;
            rc.rows = rows;
        }
        Some(Inexact(rows)) => {
            rc.cardinality = vx_cardinality::VX_CARD_ESTIMATE;
            rc.rows = rows;
        }
        None => {
            rc.cardinality = vx_cardinality::VX_CARD_UNKNOWN;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::ptr;

    use crate::data_source::vx_cardinality;
    use crate::data_source::vx_data_source_dtype;
    use crate::data_source::vx_data_source_free;
    use crate::data_source::vx_data_source_get_row_count;
    use crate::data_source::vx_data_source_new;
    use crate::data_source::vx_data_source_options;
    use crate::data_source::vx_data_source_row_count;
    use crate::dtype::vx_dtype;
    use crate::session::vx_session_free;
    use crate::session::vx_session_new;
    use crate::tests::SAMPLE_ROWS;
    use crate::tests::assert_error;
    use crate::tests::assert_no_error;
    use crate::tests::write_sample;

    #[test]
    fn test_create_invalid() {
        unsafe {
            let session = vx_session_new();
            let mut error = ptr::null_mut();

            let ds = vx_data_source_new(ptr::null_mut(), ptr::null(), &raw mut error);
            assert_error(error);
            assert!(ds.is_null());

            let ds = vx_data_source_new(session, ptr::null(), &raw mut error);
            assert_error(error);
            assert!(ds.is_null());

            let mut opts = vx_data_source_options::default();
            let ds = vx_data_source_new(session, &raw const opts, &raw mut error);
            assert_error(error);
            assert!(ds.is_null());

            opts.files = c"test.vortex".as_ptr();
            let ds = vx_data_source_new(session, &raw const opts, &raw mut error);
            assert_error(error);
            assert!(ds.is_null());

            opts.files = c"*.vortex".as_ptr();
            let ds = vx_data_source_new(session, &raw const opts, &raw mut error);
            assert_error(error);
            assert!(ds.is_null());

            vx_session_free(session);
        }
    }

    #[test]
    fn test_row_count() {
        unsafe {
            let session = vx_session_new();
            let (sample, struct_array) = write_sample(session);

            let path = CString::new(sample.path().to_str().unwrap()).unwrap();
            let opts = vx_data_source_options {
                files: path.as_ptr(),
                ..Default::default()
            };

            let mut error = ptr::null_mut();
            let ds = vx_data_source_new(session, &raw const opts, &raw mut error);
            assert_no_error(error);
            assert!(!ds.is_null());

            let dtype = vx_dtype::as_ref(vx_data_source_dtype(ds));
            assert_eq!(dtype, struct_array.dtype());

            let mut row_count = vx_data_source_row_count {
                cardinality: vx_cardinality::VX_CARD_UNKNOWN,
                rows: 0,
            };
            vx_data_source_get_row_count(ds, &raw mut row_count);
            assert_eq!(row_count.cardinality, vx_cardinality::VX_CARD_MAXIMUM);
            assert_eq!(row_count.rows, SAMPLE_ROWS as u64);

            vx_data_source_free(ds);
            vx_session_free(session);
        }
    }
}
