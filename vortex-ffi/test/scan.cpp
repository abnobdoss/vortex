// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <unistd.h>
#include <vortex.h>
#include "common.h"

namespace fs = std::filesystem;
using namespace std::string_literals;
using namespace std::string_view_literals;
using Catch::Matchers::ContainsSubstring;

struct TempPath : fs::path {
    ~TempPath() {
        fs::remove(*this);
    }
};

// StructArray { age: u8, height: u16? }
[[nodiscard]] const vx_dtype *sample_dtype() {
    vx_struct_fields_builder *builder = vx_struct_fields_builder_new();

    constexpr auto age = "age"sv;
    const vx_string *age_name = vx_string_new(age.data(), age.size());
    const vx_dtype *age_type = vx_dtype_new_primitive(PTYPE_U8, false);
    vx_struct_fields_builder_add_field(builder, age_name, age_type);

    constexpr auto height = "height"sv;
    const vx_string *height_name = vx_string_new(height.data(), height.size());
    const vx_dtype *height_type = vx_dtype_new_primitive(PTYPE_U16, true);
    vx_struct_fields_builder_add_field(builder, height_name, height_type);

    vx_struct_fields *fields = vx_struct_fields_builder_finalize(builder);
    return vx_dtype_new_struct(fields, false);
}

constexpr size_t SAMPLE_ROWS = 100;
[[nodiscard]] const vx_array *sample_array() {
    vx_validity validity = {};
    validity.type = VX_VALIDITY_NON_NULLABLE;

    vx_struct_column_builder *builder = vx_struct_column_builder_new(&validity, SAMPLE_ROWS);

    vx_error *error = nullptr;

    std::vector<uint8_t> age_buffer;
    for (uint8_t age = 0; age < SAMPLE_ROWS; ++age) {
        age_buffer.push_back(age);
    }
    const vx_array *age_array =
        vx_array_new_primitive(PTYPE_U8, age_buffer.data(), age_buffer.size(), &validity, &error);
    require_no_error(error);

    vx_struct_column_builder_add_field(builder, "age", age_array, &error);
    require_no_error(error);
    vx_array_free(age_array);

    std::vector<uint16_t> height_buffer;
    for (uint16_t height = 0; height < SAMPLE_ROWS; ++height) {
        height_buffer.push_back(1 + rand() % (height + 1));
    }
    validity.type = VX_VALIDITY_ALL_VALID;
    const vx_array *height_array =
        vx_array_new_primitive(PTYPE_U16, height_buffer.data(), height_buffer.size(), &validity, &error);
    require_no_error(error);

    vx_struct_column_builder_add_field(builder, "height", height_array, &error);
    require_no_error(error);
    vx_array_free(height_array);

    const vx_array *array = vx_struct_column_builder_finalize(builder, &error);
    require_no_error(error);
    return array;
}

[[nodiscard]] TempPath write_sample(vx_session *session, fs::path &&path) {
    REQUIRE(path.is_absolute());

    const vx_dtype *dtype = sample_dtype();

    vx_error *error = nullptr;
    vx_array_sink *sink = vx_array_sink_open_file(session, path.c_str(), dtype, &error);
    REQUIRE(sink != nullptr);
    require_no_error(error);
    vx_dtype_free(dtype);

    const vx_array *array = sample_array();
    vx_array_sink_push(sink, array, &error);
    require_no_error(error);
    vx_array_free(array);

    vx_array_sink_close(sink, &error);
    require_no_error(error);

    INFO("Written vortex file "s + path.generic_string());
    return {path};
}

TEST_CASE("Creating datasources", "[datasource]") {
    vx_session *session = vx_session_new();
    vx_error *error = nullptr;

    const vx_data_source *ds = vx_data_source_new(session, nullptr, &error);
    REQUIRE(ds == nullptr);
    REQUIRE(error != nullptr);
    vx_error_free(error);

    vx_data_source_options opts = {};
    ds = vx_data_source_new(session, &opts, &error);
    REQUIRE(ds == nullptr);
    REQUIRE(error != nullptr);
    REQUIRE_THAT(to_string(error), ContainsSubstring("opts.files"));
    vx_error_free(error);

    // First file is opened eagerly
    opts.files = "nonexistent";
    ds = vx_data_source_new(session, &opts, &error);
    REQUIRE(ds == nullptr);
    REQUIRE(error != nullptr);
    REQUIRE_THAT(to_string(error), ContainsSubstring("No such file or directory"));
    vx_error_free(error);

    opts.files = "/tmp/*.vortex";
    ds = vx_data_source_new(session, &opts, &error);
    REQUIRE(ds == nullptr);
    REQUIRE(error != nullptr);
    // TODO Object store error: Generic LocalFileSystem error: Unable to walk dir: File
    // system loop found: /dev/fd/6 points to an ancestor /
    // REQUIRE_THAT(to_string(error), ContainsSubstring("No such file or directory"));
    vx_error_free(error);

    TempPath file = write_sample(session, fs::current_path() / "empty.vortex");

    for (const char *files :
         // TODO Object store error: Generic LocalFileSystem error: Unable to walk dir: File
         // system loop found: /dev/fd/6 points to an ancestor /
         //{ file.c_str(), "*.vortex"}
         {file.c_str()}) {
        INFO("reading "s + files);
        opts.files = files;
        ds = vx_data_source_new(session, &opts, &error);
        require_no_error(error);
        REQUIRE(ds != nullptr);
        vx_data_source_free(ds);
    }

    vx_session_free(session);
}

TEST_CASE("Write file", "[datasource]") {
    vx_session *session = vx_session_new();
    TempPath path = write_sample(session, fs::current_path() / "write-file.vortex");
    vx_session_free(session);
}

TEST_CASE("Write file and read back types", "[datasource]") {
    vx_session *session = vx_session_new();
    TempPath path = write_sample(session, fs::current_path() / "write-read-types.vortex");
    vx_error *error = nullptr;

    vx_data_source_options opts = {};
    opts.files = path.c_str();

    const vx_data_source *ds = vx_data_source_new(session, &opts, &error);
    require_no_error(error);
    REQUIRE(ds != nullptr);
    vx_session_free(session);

    vx_data_source_row_count row_count = {};
    vx_data_source_get_row_count(ds, &row_count);

    CHECK(row_count.cardinality == VX_CARD_MAXIMUM);
    CHECK(row_count.rows == SAMPLE_ROWS);

    const vx_dtype *data_source_dtype = vx_data_source_dtype(ds);
    REQUIRE(vx_dtype_get_variant(data_source_dtype) == DTYPE_STRUCT);

    const vx_struct_fields *fields = vx_dtype_struct_dtype(data_source_dtype);
    const size_t len = vx_struct_fields_nfields(fields);
    REQUIRE(len == 2);

    const vx_dtype *col1_dtype = vx_struct_fields_field_dtype(fields, 0);
    const vx_string *col1_name = vx_struct_fields_field_name(fields, 0);

    REQUIRE(vx_dtype_get_variant(col1_dtype) == DTYPE_PRIMITIVE);
    REQUIRE(vx_dtype_primitive_ptype(col1_dtype) == PTYPE_U8);
    REQUIRE_FALSE(vx_dtype_is_nullable(col1_dtype));
    REQUIRE(to_string_view(col1_name) == "age");
    vx_dtype_free(col1_dtype);

    const vx_dtype *col2_dtype = vx_struct_fields_field_dtype(fields, 1);
    const vx_string *col2_name = vx_struct_fields_field_name(fields, 1);

    REQUIRE(vx_dtype_get_variant(col2_dtype) == DTYPE_PRIMITIVE);
    REQUIRE(vx_dtype_primitive_ptype(col2_dtype) == PTYPE_U16);
    REQUIRE(vx_dtype_is_nullable(col2_dtype));
    REQUIRE(to_string_view(col2_name) == "height");
    vx_dtype_free(col2_dtype);

    vx_data_source_free(ds);
}

void verify_sample_array(const vx_array *array) {
    REQUIRE(vx_array_len(array) == SAMPLE_ROWS);
    REQUIRE(vx_array_has_dtype(array, DTYPE_STRUCT));

    const vx_struct_fields *fields = vx_dtype_struct_dtype(vx_array_dtype(array));
    size_t len = vx_struct_fields_nfields(fields);
    REQUIRE(len == 2);

    const vx_dtype *age_dtype = vx_struct_fields_field_dtype(fields, 0);
    REQUIRE(vx_dtype_get_variant(age_dtype) == DTYPE_PRIMITIVE);
    REQUIRE(vx_dtype_primitive_ptype(age_dtype) == PTYPE_U8);
    vx_dtype_free(age_dtype);
    const vx_string *age_name = vx_struct_fields_field_name(fields, 0);
    REQUIRE(to_string_view(age_name) == "age");

    const vx_dtype *height_dtype = vx_struct_fields_field_dtype(fields, 1);
    REQUIRE(vx_dtype_get_variant(height_dtype) == DTYPE_PRIMITIVE);
    REQUIRE(vx_dtype_primitive_ptype(height_dtype) == PTYPE_U16);
    vx_dtype_free(height_dtype);
    const vx_string *height_name = vx_struct_fields_field_name(fields, 1);
    REQUIRE(to_string_view(height_name) == "height");

    vx_error *error = nullptr;
    vx_validity validity = {};
    vx_array_get_validity(array, &validity, &error);
    require_no_error(error);
    REQUIRE(validity.type == VX_VALIDITY_NON_NULLABLE);

    const vx_array *age_field = vx_array_get_field(array, 0, &error);
    require_no_error(error);
    REQUIRE(vx_array_has_dtype(age_field, DTYPE_PRIMITIVE));
    REQUIRE(vx_dtype_primitive_ptype(vx_array_dtype(age_field)) == PTYPE_U8);
    REQUIRE(vx_array_len(age_field) == SAMPLE_ROWS);
    for (size_t i = 0; i < SAMPLE_ROWS; ++i) {
        REQUIRE(vx_array_get_u8(age_field, i) == i);
    }
    vx_array_free(age_field);

    const vx_array *height_field = vx_array_get_field(array, 1, &error);
    require_no_error(error);
    REQUIRE(vx_array_has_dtype(height_field, DTYPE_PRIMITIVE));
    REQUIRE(vx_dtype_primitive_ptype(vx_array_dtype(height_field)) == PTYPE_U16);
    REQUIRE(vx_array_len(height_field) == SAMPLE_ROWS);
    for (size_t i = 0; i < SAMPLE_ROWS; ++i) {
        REQUIRE(vx_array_get_u16(height_field, i) > 0);
    }
    vx_array_free(height_field);

    REQUIRE(vx_array_get_field(array, 2, &error) == nullptr);
    REQUIRE(error != nullptr);
    vx_error_free(error);
}

TEST_CASE("Requesting scans", "[datasource]") {
    vx_session *session = vx_session_new();
    TempPath path = write_sample(session, fs::current_path() / "write-file2.vortex");

    vx_data_source_options ds_options = {};
    ds_options.files = path.c_str();

    vx_error *error = nullptr;
    const vx_data_source *ds = vx_data_source_new(session, &ds_options, &error);
    require_no_error(error);
    REQUIRE(ds != nullptr);

    vx_scan *scan = vx_data_source_scan(ds, nullptr, nullptr, &error);
    require_no_error(error);
    REQUIRE(scan != nullptr);
    vx_scan_free(scan);

    vx_scan_options options = {};
    options.max_threads = 1;
    scan = vx_data_source_scan(ds, &options, nullptr, &error);
    require_no_error(error);
    REQUIRE(scan != nullptr);
    vx_scan_free(scan);

    vx_data_source_free(ds);
    vx_session_free(session);
}

TEST_CASE("Basic scan", "[datasource]") {
    vx_session *session = vx_session_new();
    TempPath path = write_sample(session, fs::current_path() / "basic-scan.vortex");
    vx_error *error = nullptr;

    vx_data_source_options ds_options = {};
    ds_options.files = path.c_str();

    const vx_data_source *ds = vx_data_source_new(session, &ds_options, &error);
    require_no_error(error);
    REQUIRE(ds != nullptr);

    vx_scan *scan = vx_data_source_scan(ds, nullptr, nullptr, &error);
    require_no_error(error);
    REQUIRE(scan != nullptr);

    vx_partition *partition = vx_scan_next(scan, &error);
    require_no_error(error);

    vx_estimate estimate = {};
    vx_partition_row_count(partition, &estimate, &error);
    require_no_error(error);
    REQUIRE(estimate.type == VX_ESTIMATE_EXACT);
    REQUIRE(estimate.estimate == SAMPLE_ROWS);

    REQUIRE(vx_scan_next(scan, &error) == nullptr);
    require_no_error(error);

    const vx_array *array = vx_partition_next(partition, &error);
    require_no_error(error);
    REQUIRE(array != nullptr);

    REQUIRE(vx_partition_next(partition, &error) == nullptr);
    require_no_error(error);

    verify_sample_array(array);

    vx_array_free(array);
    vx_partition_free(partition);
    vx_scan_free(scan);

    vx_data_source_free(ds);
    vx_session_free(session);
}
