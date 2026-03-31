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

constexpr size_t SAMPLE_ROWS = 0;
[[nodiscard]] TempPath write_sample(vx_session *session, fs::path &&path) {
    REQUIRE(path.is_absolute());
    vx_struct_fields_builder *builder = vx_struct_fields_builder_new();

    {
        constexpr auto col1 = "col1"sv;
        const vx_string *col1_name = vx_string_new(col1.data(), col1.size());
        const vx_dtype *col1_dtype = vx_dtype_new_primitive(PTYPE_U8, false);
        vx_struct_fields_builder_add_field(builder, col1_name, col1_dtype);
    }
    {
        constexpr auto col2 = "col2"sv;
        const vx_string *col2_name = vx_string_new(col2.data(), col2.size());
        const vx_dtype *col2_dtype = vx_dtype_new_utf8(true);
        vx_struct_fields_builder_add_field(builder, col2_name, col2_dtype);
    }

    vx_struct_fields *fields = vx_struct_fields_builder_finalize(builder);
    const vx_dtype *file_dtype = vx_dtype_new_struct(fields, false);

    vx_error *error = nullptr;
    vx_array_sink *sink = vx_array_sink_open_file(session, path.c_str(), file_dtype, &error);
    REQUIRE(sink != nullptr);
    require_no_error(error);
    vx_dtype_free(file_dtype);

    for (size_t i = 0; i < SAMPLE_ROWS; ++i) {
        //const vx_array* array = vx_array_new_primitive();
        //vx_array_sink_push(sink, array, &error);
        //require_no_error(error);
        //vx_array_free(array);
    }

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
    REQUIRE(to_string_view(col1_name) == "col1");
    vx_dtype_free(col1_dtype);

    const vx_dtype *col2_dtype = vx_struct_fields_field_dtype(fields, 1);
    const vx_string *col2_name = vx_struct_fields_field_name(fields, 1);

    REQUIRE(vx_dtype_get_variant(col2_dtype) == DTYPE_UTF8);
    REQUIRE(vx_dtype_is_nullable(col2_dtype));
    REQUIRE(to_string_view(col2_name) == "col2");
    vx_dtype_free(col2_dtype);

    vx_data_source_free(ds);
}

//TEST_CASE("Write file and read back", "[datasource]") {
//    vx_session *session = vx_session_new();
//    TempPath path = write_sample(session, fs::current_path() / "write-read.vortex");
//    vx_error *error = nullptr;
//
//    vx_data_source_options ds_options = {};
//    ds_options.files = path.c_str();
//
//    const vx_data_source *ds = vx_data_source_new(session, &ds_options, &error);
//    require_no_error(error);
//    REQUIRE(ds != nullptr);
//
//    vx_scan *scan = vx_data_source_scan(ds, nullptr, nullptr, &error);
//    require_no_error(error);
//    REQUIRE(scan != nullptr);
//    vx_scan_free(scan);
//
//    vx_scan_options scan_options = {};
//    scan = vx_data_source_scan(ds, &scan_options, nullptr, &error);
//    require_no_error(error);
//    REQUIRE(scan != nullptr);
//    vx_scan_free(scan);
//
//    vx_data_source_free(ds);
//    vx_session_free(session);
//}
