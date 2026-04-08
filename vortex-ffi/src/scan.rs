#![allow(non_camel_case_types)]

use core::slice;
use std::ffi::c_int;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_schema::ArrowError;
use arrow_schema::DataType;
use futures::StreamExt;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::VortexSessionExecute;
use vortex::array::arrow::ArrowArrayExecutor;
use vortex::array::expr::stats::Precision;
use vortex::array::stream::SendableArrayStream;
use vortex::buffer::Buffer;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::expr::root;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::scan::arrow::RecordBatchIteratorAdapter;
use vortex::scan::DataSourceScan;
use vortex::scan::Partition;
use vortex::scan::PartitionStream;
use vortex::scan::ScanRequest;
use vortex::scan::selection::Selection;

use crate::RUNTIME;
use crate::array::vx_array;
use crate::data_source::vx_data_source;
use crate::dtype::vx_dtype;
use crate::error::try_or;
use crate::error::try_or_default;
use crate::error::vx_error;
use crate::expression::vx_expression;
use crate::session::vx_session;

pub enum VxScanState {
    Pending(Box<dyn DataSourceScan>),
    Started(PartitionStream),
    Finished,
}
pub type VxScan = Mutex<VxScanState>;
crate::box_wrapper!(VxScan, vx_scan);

pub enum VxPartitionScan {
    Pending(Box<dyn Partition>),
    Started(SendableArrayStream),
    Finished,
}
crate::box_wrapper!(
    /// A Partition is a unit of work for a worker thread from which you can
    /// get vx_arrays.
    VxPartitionScan,
    vx_partition);

#[repr(C)]
// We parse Selection from vx_scan_selection[_include], so we don't need
// to instantiate VX_S_* items directly.
#[allow(dead_code)]
#[cfg_attr(test, derive(Default))]
pub enum vx_scan_selection_include {
    #[cfg_attr(test, default)]
    VX_S_INCLUDE_ALL = 0,
    VX_S_INCLUDE_RANGE = 1,
    VX_S_EXCLUDE_RANGE = 2,
}

#[repr(C)]
#[cfg_attr(test, derive(Default))]
pub struct vx_scan_selection {
    pub idx: *const u64,
    pub idx_len: usize,
    pub include: vx_scan_selection_include,
}

#[repr(C)]
#[cfg_attr(test, derive(Default))]
pub struct vx_scan_options {
    pub projection: *const vx_expression,
    pub filter: *const vx_expression,
    pub row_range_begin: u64,
    pub row_range_end: u64,
    pub selection: vx_scan_selection,
    pub limit: u64,
    pub max_threads: u64,
    pub ordered: c_int,
}

#[repr(C)]
pub enum vx_estimate_boundary {
    VX_ESTIMATE_UNKNOWN = 0,
    VX_ESTIMATE_EXACT = 1,
    VX_ESTIMATE_INEXACT = 2,
}

#[repr(C)]
pub struct vx_estimate {
    estimate: u64,
    r#type: vx_estimate_boundary,
}

fn scan_request(opts: *const vx_scan_options) -> VortexResult<ScanRequest> {
    if opts.is_null() {
        return Ok(ScanRequest::default());
    }
    let opts = unsafe { &*opts };

    let projection = if opts.projection.is_null() {
        root()
    } else {
        vx_expression::as_ref(opts.projection).clone()
    };

    let filter = if opts.filter.is_null() {
        None
    } else {
        Some(vx_expression::as_ref(opts.filter).clone())
    };

    let selection = &opts.selection;
    let selection = match selection.include {
        vx_scan_selection_include::VX_S_INCLUDE_ALL => Selection::All,
        vx_scan_selection_include::VX_S_INCLUDE_RANGE => {
            let buf = unsafe { slice::from_raw_parts(selection.idx, selection.idx_len) };
            let buf = Buffer::copy_from(buf);
            Selection::IncludeByIndex(buf)
        }
        vx_scan_selection_include::VX_S_EXCLUDE_RANGE => {
            let buf = unsafe { slice::from_raw_parts(selection.idx, selection.idx_len) };
            let buf = Buffer::copy_from(buf);
            Selection::ExcludeByIndex(buf)
        }
    };

    let ordered = opts.ordered == 1;

    let start = opts.row_range_begin;
    let end = opts.row_range_end;
    let row_range = (start > 0 || end > 0).then_some(Range { start, end });

    let limit = (opts.limit != 0).then_some(opts.limit);

    Ok(ScanRequest {
        projection,
        filter,
        row_range,
        selection,
        ordered,
        limit,
    })
}

fn write_estimate<T: Into<u64>>(estimate: Option<Precision<T>>, out: &mut vx_estimate) {
    match estimate {
        Some(Precision::Exact(value)) => {
            out.r#type = vx_estimate_boundary::VX_ESTIMATE_EXACT;
            out.estimate = value.into();
        }
        Some(Precision::Inexact(value)) => {
            out.r#type = vx_estimate_boundary::VX_ESTIMATE_INEXACT;
            out.estimate = value.into();
        }
        None => {
            out.r#type = vx_estimate_boundary::VX_ESTIMATE_UNKNOWN;
        }
    }
}

#[unsafe(no_mangle)]
/// Create a new owned data source scan which must be freed by the caller.
/// Scan can be consumed only once.
/// Returns NULL and sets err on error.
/// options may not be NULL.
/// If estimate is not NULL, return estimate on the number of partitions.
pub unsafe extern "C-unwind" fn vx_data_source_scan(
    data_source: *const vx_data_source,
    options: *const vx_scan_options,
    estimate: *mut vx_estimate,
    err: *mut *mut vx_error,
) -> *mut vx_scan {
    try_or(err, ptr::null_mut(), || {
        let request = scan_request(options)?;
        RUNTIME.block_on(async {
            let scan = vx_data_source::as_ref(data_source).scan(request).await?;
            if !estimate.is_null() {
                write_estimate(
                    scan.partition_count().map(|x| match x {
                        Precision::Exact(v) => Precision::Exact(v as u64),
                        Precision::Inexact(v) => Precision::Inexact(v as u64),
                    }),
                    unsafe { &mut *estimate },
                );
            }
            Ok(vx_scan::new(Box::new(Mutex::new(VxScanState::Pending(
                scan,
            )))))
        })
    })
}

/// Return an owned dtype of the scan.
/// On error, returns NULL and sets err.
/// You can't request a dtype of a scan that's already started.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scan_dtype(
    scan: *const vx_scan,
    err: *mut *mut vx_error,
) -> *const vx_dtype {
    try_or(err, ptr::null(), || {
        let scan = vx_scan::as_ref(scan).lock().unwrap();
        let VxScanState::Pending(ref scan) = *scan else {
            vortex_bail!("can't get dtype after scan is started");
        };
        Ok(vx_dtype::new(Arc::new(scan.dtype().clone())))
    })
}

/// Get next owned partition out of a scan request.
/// Caller must free this partition using vx_partition_free.
/// This method is thread-safe.
/// If using in a sync multi-thread runtime, users are encouraged to create a
/// worker thread per partition.
/// Returns NULL and doesn't set err on exhaustion.
/// Returns NULL and sets err on error.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scan_next(
    scan: *mut vx_scan,
    err: *mut *mut vx_error,
) -> *mut vx_partition {
    let scan = vx_scan::as_mut(scan);
    let mut scan = scan.lock().expect("failed to lock mutex");
    let scan = &mut *scan;
    unsafe {
        let ptr = scan as *mut VxScanState;

        let on_finish = || -> VortexResult<*mut vx_partition> {
            ptr::write(ptr, VxScanState::Finished);
            Ok(ptr::null_mut())
        };

        let on_stream = |mut stream: PartitionStream| -> VortexResult<*mut vx_partition> {
            match RUNTIME.block_on(stream.next()) {
                Some(partition) => {
                    let partition = VxPartitionScan::Pending(partition?);
                    let partition = vx_partition::new(Box::new(partition));
                    ptr::write(ptr, VxScanState::Started(stream));
                    Ok(partition)
                }
                None => on_finish(),
            }
        };

        let owned = ptr::read(ptr);
        try_or_default(err, || match owned {
            VxScanState::Pending(scan) => on_stream(scan.partitions()),
            VxScanState::Started(stream) => on_stream(stream),
            VxScanState::Finished => on_finish(),
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_partition_row_count(
    partition: *const vx_partition,
    count: *mut vx_estimate,
    err: *mut *mut vx_error,
) -> c_int {
    try_or(err, 1, || {
        let partition = vx_partition::as_ref(partition);
        let VxPartitionScan::Pending(partition) = partition else {
            vortex_bail!("Can't get partition row count: partition already being consumed");
        };
        write_estimate(partition.row_count(), unsafe { &mut *count });
        Ok(0)
    })
}

/// Scan partition contents to ArrowArrayStream. This function consumes
/// partition fully. Subsequent calls to vx_partition_scan_arrow or
/// vx_partition_next are undefined behaviour.
///
/// If this function errors, you can't free or reuse partition.
///
/// Caller still needs to free partition after calling this function.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_partition_scan_arrow(
    session: *const vx_session,
    partition: *mut vx_partition,
    stream: *mut FFI_ArrowArrayStream,
    err: *mut *mut vx_error,
) -> c_int {
    return 1;
    //try_or(err, 1, || {
    //    let ptr = partition as *mut VxPartitionScan;
    //    let owned = unsafe { ptr::read(ptr) };
    //    let partition = match owned {
    //        VxPartitionScan::Pending(partition) => partition,
    //        _ => vortex_bail!(
    //            "Can't consume partition into ArrowArrayStream: partition already being consumed"
    //        ),
    //    };
    //    unsafe { ptr::write(ptr, VxPartitionScan::Finished); };
    //    let array_stream = partition.execute()?;
    //    let dtype = array_stream.dtype();

    //    let schema = dtype.to_arrow_schema()?;
    //    let schema = Arc::new(schema);
    //    let data_type = DataType::Struct(schema.fields().clone());

    //    let session = vx_session::as_ref(session);

    //    let on_chunk = move |chunk: VortexResult<ArrayRef>| -> VortexResult<RecordBatch> {
    //        let chunk: ArrayRef = chunk?;
    //        let mut ctx: ExecutionCtx = session.create_execution_ctx();
    //        let arrow = chunk.execute_arrow(Some(&data_type), &mut ctx)?;
    //        Ok(RecordBatch::from(arrow.as_struct().clone()))
    //    };

    //    let iter: Result<RecordBatch, ArrowError> = array_stream
    //        .map(on_chunk)
    //        .into_iter(&*RUNTIME)?
    //        .map(|result| result.map_err(|e| ArrowError::ExternalError(Box::new(e))));

    //    let reader = RecordBatchIteratorAdapter::new(iter, schema);
    //    let arrow_stream = FFI_ArrowArrayStream::new(Box::new(reader));
    //    unsafe { ptr::write(stream, arrow_stream); };
    //    Ok(0)
    //})
}

/// Get next vx_array out of this partition.
/// Thread-unsafe.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_partition_next(
    partition: *mut vx_partition,
    err: *mut *mut vx_error,
) -> *const vx_array {
    let partition = vx_partition::as_mut(partition);
    unsafe {
        let ptr = partition as *mut VxPartitionScan;

        let on_finish = || -> VortexResult<*const vx_array> {
            ptr::write(ptr, VxPartitionScan::Finished);
            Ok(ptr::null_mut())
        };

        let on_stream = |mut stream: SendableArrayStream| -> VortexResult<*const vx_array> {
            match RUNTIME.block_on(stream.next()) {
                Some(array) => {
                    let array = vx_array::new(Arc::new(array?));
                    ptr::write(ptr, VxPartitionScan::Started(stream));
                    Ok(array)
                }
                None => on_finish(),
            }
        };

        let owned = ptr::read(ptr);
        try_or_default(err, || match owned {
            VxPartitionScan::Pending(partition) => on_stream(partition.execute()?),
            VxPartitionScan::Started(stream) => on_stream(stream),
            VxPartitionScan::Finished => on_finish(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::ptr;

    use vortex::array::arrays::StructArray;
    use vortex::expr::lit;
    use vortex_array::assert_arrays_eq;

    use crate::array::vx_array;
    use crate::array::vx_array_free;
    use crate::data_source::vx_data_source_free;
    use crate::data_source::vx_data_source_new;
    use crate::data_source::vx_data_source_options;
    use crate::expression::vx_binary_operator;
    use crate::expression::vx_expression;
    use crate::expression::vx_expression_binary;
    use crate::expression::vx_expression_free;
    use crate::expression::vx_expression_get_item;
    use crate::expression::vx_expression_root;
    use crate::scan::vx_data_source_scan;
    use crate::scan::vx_estimate;
    use crate::scan::vx_partition_free;
    use crate::scan::vx_partition_next;
    use crate::scan::vx_partition_row_count;
    use crate::scan::vx_scan_free;
    use crate::scan::vx_scan_next;
    use crate::scan::vx_scan_options;
    use crate::scan::vx_scan_selection_include;
    use crate::session::vx_session_free;
    use crate::session::vx_session_new;
    use crate::tests::assert_no_error;
    use crate::tests::write_sample;

    /// Perform a scan with options over a sample file, return read array and
    /// original generated array for the sample file.
    fn scan(options: *const vx_scan_options) -> (*const vx_array, StructArray) {
        unsafe {
            let session = vx_session_new();
            let (sample, struct_array) = write_sample(session);
            let path = CString::new(sample.path().to_str().unwrap()).unwrap();
            let ds_options = vx_data_source_options {
                files: path.as_ptr(),
                ..Default::default()
            };

            let mut error = ptr::null_mut();
            let ds = vx_data_source_new(session, &raw const ds_options, &raw mut error);
            assert_no_error(error);
            assert!(!ds.is_null());

            let mut error = ptr::null_mut();
            let scan = vx_data_source_scan(ds, options, ptr::null_mut(), &raw mut error);
            assert_no_error(error);
            assert!(!scan.is_null());

            let partition = vx_scan_next(scan, &raw mut error);
            assert_no_error(error);
            assert!(!partition.is_null());

            let array = vx_partition_next(partition, &raw mut error);
            assert_no_error(error);
            assert!(!array.is_null());

            assert!(vx_partition_next(partition, &raw mut error).is_null());
            assert_no_error(error);
            assert!(vx_partition_next(partition, &raw mut error).is_null());
            assert_no_error(error);

            vx_partition_free(partition);
            vx_scan_free(scan);
            vx_data_source_free(ds);
            vx_session_free(session);

            (array, struct_array)
        }
    }

    #[test]
    fn test_no_options() {
        let (array, struct_array) = scan(ptr::null());
        assert_arrays_eq!(vx_array::as_ref(array), struct_array);
        unsafe { vx_array_free(array) };
    }

    #[test]
    fn test_project_all() {
        let opts = vx_scan_options::default();
        let (array, struct_array) = scan(&raw const opts);
        assert_arrays_eq!(vx_array::as_ref(array), struct_array);
        unsafe { vx_array_free(array) };
    }

    #[test]
    fn test_project_single_field() {
        unsafe {
            let root = vx_expression_root();
            let mut opts = vx_scan_options::default();

            for (field, c_field) in [("age", c"age"), ("height", c"height"), ("name", c"name")] {
                let field_expr = vx_expression_get_item(c_field.as_ptr(), root);
                assert!(!field_expr.is_null());
                opts.projection = field_expr;
                let (array, struct_array) = scan(&raw const opts);
                assert_arrays_eq!(
                    vx_array::as_ref(array),
                    struct_array.unmasked_field_by_name(field).unwrap()
                );
                vx_array_free(array);
                vx_expression_free(field_expr);
            }
            vx_expression_free(root);
        }
    }

    #[test]
    fn test_project_sum() {
        unsafe {
            let root = vx_expression_root();
            let mut opts = vx_scan_options::default();

            let expr_age = vx_expression_get_item(c"age".as_ptr(), root);
            let expr_height = vx_expression_get_item(c"height".as_ptr(), root);
            let expr_sum =
                vx_expression_binary(vx_binary_operator::VX_OPERATOR_ADD, expr_age, expr_height);

            opts.projection = expr_sum;
            let (array, struct_array) = scan(&raw const opts);
            //assert_arrays_eq!(
            //    vx_array::as_ref(array),
            //    struct_array.unmasked_field_by_name(field).unwrap()
            //);
            vx_array_free(array);

            vx_expression_free(expr_age);
            vx_expression_free(expr_height);
            vx_expression_free(expr_sum);
            vx_expression_free(root);
        }
    }

    #[test]
    fn test_filter() {
        unsafe {
            let root = vx_expression_root();
            let age_expr = vx_expression_get_item(c"age".as_ptr(), root);
            let lit_100 = vx_expression::new(Box::new(lit(100u64)));
            let filter =
                vx_expression_binary(vx_binary_operator::VX_OPERATOR_GTE, age_expr, lit_100);

            let mut opts = vx_scan_options::default();
            opts.filter = filter;
            let (array, _) = scan(&raw const opts);
            assert_eq!(vx_array::as_ref(array).len(), 100);

            vx_array_free(array);
            vx_expression_free(filter);
            vx_expression_free(age_expr);
            vx_expression_free(lit_100);
            vx_expression_free(root);
        }
    }

    #[test]
    fn test_filter_project() {
        unsafe {
            let root = vx_expression_root();
            let age_expr = vx_expression_get_item(c"age".as_ptr(), root);
            let lit_100 = vx_expression::new(Box::new(lit(100u64)));
            let filter =
                vx_expression_binary(vx_binary_operator::VX_OPERATOR_GTE, age_expr, lit_100);
            let age_proj = vx_expression_get_item(c"age".as_ptr(), root);

            let mut opts = vx_scan_options::default();
            opts.filter = filter;
            opts.projection = age_proj;
            let (array, _) = scan(&raw const opts);
            assert_eq!(vx_array::as_ref(array).len(), 100);

            vx_array_free(array);
            vx_expression_free(filter);
            vx_expression_free(age_expr);
            vx_expression_free(lit_100);
            vx_expression_free(age_proj);
            vx_expression_free(root);
        }
    }

    #[test]
    fn test_row_range() {
        let mut opts = vx_scan_options::default();
        opts.row_range_begin = 50;
        opts.row_range_end = 100;
        let (array, _) = scan(&raw const opts);
        assert_eq!(vx_array::as_ref(array).len(), 50);
        unsafe { vx_array_free(array) };
    }

    #[test]
    fn test_selection() {
        let indices = [0u64, 50, 100, 150, 199];
        let mut opts = vx_scan_options::default();
        opts.selection.idx = indices.as_ptr();
        opts.selection.idx_len = indices.len();
        opts.selection.include = vx_scan_selection_include::VX_S_INCLUDE_RANGE;
        let (array, _) = scan(&raw const opts);
        assert_eq!(vx_array::as_ref(array).len(), indices.len());
        unsafe { vx_array_free(array) };
    }

    #[test]
    fn test_limit() {
        let mut opts = vx_scan_options::default();
        opts.limit = 50;
        let (array, _) = scan(&raw const opts);
        assert_eq!(vx_array::as_ref(array).len(), 50);
        unsafe { vx_array_free(array) };
    }

    #[test]
    fn test_ordered() {
        let mut opts = vx_scan_options::default();
        opts.ordered = 1;
        let (array, struct_array) = scan(&raw const opts);
        assert_arrays_eq!(vx_array::as_ref(array), struct_array);
        unsafe { vx_array_free(array) };
    }

    #[test]
    fn test_row_count() {
        unsafe {
            let session = vx_session_new();
            let (sample, _) = write_sample(session);
            let path = CString::new(sample.path().to_str().unwrap()).unwrap();
            let ds_options = vx_data_source_options {
                files: path.as_ptr(),
                ..Default::default()
            };

            let mut error = ptr::null_mut();
            let ds = vx_data_source_new(session, &raw const ds_options, &raw mut error);
            assert_no_error(error);

            let mut error = ptr::null_mut();
            let scan_ptr = vx_data_source_scan(ds, ptr::null(), ptr::null_mut(), &raw mut error);
            assert_no_error(error);

            let mut error = ptr::null_mut();
            let partition = vx_scan_next(scan_ptr, &raw mut error);
            assert_no_error(error);
            assert!(!partition.is_null());

            let mut count: vx_estimate = std::mem::zeroed();
            let result = vx_partition_row_count(partition, &raw mut count, &raw mut error);
            assert_no_error(error);
            assert_eq!(result, 0);

            vx_partition_free(partition);
            vx_scan_free(scan_ptr);
            vx_data_source_free(ds);
            vx_session_free(session);
        }
    }
}
