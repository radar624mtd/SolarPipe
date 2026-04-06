"""
SolarPipe gRPC sidecar stub — Phase 2 deterministic mock.

Returns fixed predictions from Arrow IPC files. No PyTorch.
Validates Arrow IPC schema enforcement end-to-end (ADR-011, RULE-063).

Usage:
    python solarpipe_stub.py [--port 50051]

Requirements (no virtualenv needed for stub):
    pip install grpcio grpcio-tools pyarrow

Generate gRPC code from proto:
    python -m grpc_tools.protoc \
        -I. --python_out=. --grpc_python_out=. \
        solarpipe.proto
"""

import argparse
import logging
import struct
import sys
import time
from concurrent import futures
from pathlib import Path

import grpc
import pyarrow as pa
import pyarrow.ipc as pa_ipc

# Generated gRPC stubs (created by grpc_tools.protoc from solarpipe.proto)
# These imports will fail until proto compilation — expected in Phase 2 setup.
try:
    import solarpipe_pb2
    import solarpipe_pb2_grpc
    _PROTO_AVAILABLE = True
except ImportError:
    _PROTO_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": %(message)s}',
)
logger = logging.getLogger("solarpipe.stub")

# Deterministic prediction seed: arrival_time = 48h for all inputs.
# Unit tests assert this exact value.
STUB_PREDICTION_HOURS = 48.0

# Schema enforcement: all prediction columns must be float32 (RULE-063).
REQUIRED_PREDICTION_SCHEMA = pa.schema([
    pa.field("prediction", pa.float32()),
])


def _read_arrow_ipc(path: str) -> pa.Table:
    """Read an Arrow IPC file and validate float32 column types."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arrow IPC file not found: {path}")

    with pa_ipc.open_file(pa.memory_map(str(p), "r")) as reader:
        table = reader.read_all()

    # Enforce float32 schema for all numeric columns (RULE-063)
    for col in table.schema:
        if pa.types.is_floating(col.type) and col.type != pa.float32():
            raise TypeError(
                f"Arrow IPC schema violation: column '{col.name}' has type {col.type}, "
                f"expected float32. Enforce pa.float32() schema at write time (RULE-063)."
            )

    return table


def _write_predictions_arrow_ipc(output_path: str, row_count: int) -> None:
    """Write deterministic stub predictions as Arrow IPC (float32, RULE-063)."""
    values = pa.array(
        [STUB_PREDICTION_HOURS] * row_count,
        type=pa.float32(),
    )
    table = pa.table({"prediction": values})

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with pa_ipc.new_file(str(p), REQUIRED_PREDICTION_SCHEMA) as writer:
        writer.write_table(table)


class PythonTrainerStub:
    """Deterministic gRPC stub implementing PythonTrainer service."""

    def Train(self, request, context):
        logger.info('"Train called: stage=%s model=%s"', request.stage_name, request.model_type)

        try:
            table = _read_arrow_ipc(request.arrow_ipc_path)
            row_count = table.num_rows
        except Exception as exc:
            return solarpipe_pb2.TrainResponse(error_message=str(exc))

        model_id = f"stub_{request.stage_name}_v1"
        logger.info('"Train complete: model_id=%s rows=%d"', model_id, row_count)

        return solarpipe_pb2.TrainResponse(
            model_id=model_id,
            final_train_loss=0.05,
            final_val_loss=0.06,
            epochs_completed=1,
        )

    def StreamTrain(self, request, context):
        logger.info('"StreamTrain called: stage=%s"', request.stage_name)

        try:
            table = _read_arrow_ipc(request.arrow_ipc_path)
        except Exception as exc:
            yield solarpipe_pb2.TrainProgress(error_message=str(exc), is_final=True)
            return

        # Emit two progress messages then final
        for epoch in range(1, 3):
            yield solarpipe_pb2.TrainProgress(
                epoch=epoch,
                train_loss=0.10 / epoch,
                val_loss=0.12 / epoch,
                is_final=False,
            )

        model_id = f"stub_{request.stage_name}_v1"
        yield solarpipe_pb2.TrainProgress(
            epoch=2,
            train_loss=0.05,
            val_loss=0.06,
            is_final=True,
            model_id=model_id,
        )

    def Predict(self, request, context):
        logger.info('"Predict called: model_id=%s"', request.model_id)

        try:
            table = _read_arrow_ipc(request.arrow_ipc_path)
            row_count = table.num_rows
            _write_predictions_arrow_ipc(request.output_arrow_ipc_path, row_count)
        except Exception as exc:
            return solarpipe_pb2.PredictResponse(error_message=str(exc))

        logger.info('"Predict complete: rows=%d output=%s"', row_count, request.output_arrow_ipc_path)
        return solarpipe_pb2.PredictResponse(
            output_arrow_ipc_path=request.output_arrow_ipc_path,
            row_count=row_count,
        )

    def ExportOnnx(self, request, context):
        logger.info('"ExportOnnx called: model_id=%s"', request.model_id)
        # Stub: write a minimal 1-byte placeholder
        p = Path(request.onnx_output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00" * 8)
        return solarpipe_pb2.ExportOnnxResponse(
            onnx_output_path=request.onnx_output_path,
        )


def serve(port: int = 50051) -> None:
    if not _PROTO_AVAILABLE:
        logger.error(
            '"Proto stubs not compiled. Run: python -m grpc_tools.protoc -I. '
            '--python_out=. --grpc_python_out=. solarpipe.proto"'
        )
        sys.exit(1)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    solarpipe_pb2_grpc.add_PythonTrainerServicer_to_server(PythonTrainerStub(), server)

    # RULE-062: Bind to 0.0.0.0, not localhost, for container compatibility.
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    logger.info('"SolarPipe gRPC stub listening on 0.0.0.0:%d"', port)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SolarPipe gRPC stub")
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()
    serve(args.port)
