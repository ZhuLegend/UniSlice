# Use an existing CANN image validated for the host driver and device.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Supply a compatible ONNX Runtime/CANN distribution in the build context.
COPY onnx/ /opt/unislice/onnxruntime/lib/
ENV ORT_DYLIB_PATH=/opt/unislice/onnxruntime/lib/libonnxruntime.so
ENV LD_LIBRARY_PATH=/opt/unislice/onnxruntime/lib:${LD_LIBRARY_PATH}

# Preserve the base image environment; the node supplies the worker command.
ENTRYPOINT []
CMD []
