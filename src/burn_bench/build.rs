fn main() {
    // burn-import's ONNX loader generates Rust code from the .onnx
    // file at build time. The generated module is included below.
    //
    // The model file must be present at:
    //   $CARGO_MANIFEST_DIR/../../models/mobilenetv2-7.onnx
    //
    // See: https://burn.dev/docs/burn-import/

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../models/mobilenetv2-7.onnx");

    // Register ONNX model for compile-time conversion
    match burn_import::onnx::ONNXGraph::new(
        "../../models/mobilenetv2-7.onnx",
    ) {
        Ok(graph) => {
            graph.into_register().unwrap();
        }
        Err(e) => {
            panic!("Failed to load ONNX model: {e}");
        }
    }
}
