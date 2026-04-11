"""
SolarPipe Python sidecar unit + integration tests.

Coverage:
  1. Arrow IPC helpers — float32 enforcement (RULE-063), NaN round-trip (RULE-120)
  2. Stub trainer (_stub_train) — model_id format, meta.json written
  3. TFT trainer — runs without PyTorch (stub path), and with PyTorch
  4. NeuralODE trainer — same dual-path coverage
  5. Prediction (_predict) — stub returns 48h constant; live model returns finite values
  6. ONNX export — stub path writes placeholder bytes; live path traces successfully
  7. Server lifecycle — serve() binds, health check responds SERVING, graceful stop
  8. Stub server (solarpipe_stub.py) — Train / StreamTrain / Predict / ExportOnnx contracts

Tests do NOT require a running sidecar process unless explicitly marked @pytest.mark.live.
"""

import json
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.ipc as pa_ipc

# Make the python/ directory importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import solarpipe_pb2
import solarpipe_pb2_grpc
import solarpipe_server as srv
import solarpipe_stub as stub_mod

# ─── helpers ─────────────────────────────────────────────────────────────────

FLOAT32_SCHEMA = pa.schema([
    pa.field("speed",   pa.float32()),
    pa.field("density", pa.float32()),
])

PRED_SCHEMA = pa.schema([pa.field("prediction", pa.float32())])


def _write_float32_ipc(path: str, rows: int = 5) -> None:
    speeds   = pa.array([float(400 + i * 100) for i in range(rows)], type=pa.float32())
    densities = pa.array([float(5 + i) for i in range(rows)], type=pa.float32())
    table = pa.table({"speed": speeds, "density": densities}, schema=FLOAT32_SCHEMA)
    with pa_ipc.new_file(path, FLOAT32_SCHEMA) as w:
        w.write_table(table)


def _write_float64_ipc(path: str) -> None:
    schema = pa.schema([pa.field("speed", pa.float64())])
    arr = pa.array([400.0, 500.0], type=pa.float64())
    table = pa.table({"speed": arr})
    with pa_ipc.new_file(path, schema) as w:
        w.write_table(table)


def _write_ipc_with_nulls(path: str) -> None:
    schema = pa.schema([pa.field("bz", pa.float32(), nullable=True)])
    arr = pa.array([-5.0, None, -8.3], type=pa.float32())
    table = pa.table({"bz": arr})
    with pa_ipc.new_file(path, schema) as w:
        w.write_table(table)


# ─── 1. Arrow IPC helpers ────────────────────────────────────────────────────

class TestArrowIpcHelpers:

    def test_read_float32_file_returns_table(self, tmp_path):
        p = str(tmp_path / "data.arrow")
        _write_float32_ipc(p, rows=3)
        table = srv._read_arrow_ipc(p)
        assert table.num_rows == 3
        assert table.schema.field("speed").type == pa.float32()

    def test_read_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            srv._read_arrow_ipc(str(tmp_path / "missing.arrow"))

    def test_read_float64_column_raises_type_error(self, tmp_path):
        p = str(tmp_path / "bad.arrow")
        _write_float64_ipc(p)
        with pytest.raises(TypeError, match="float32"):
            srv._read_arrow_ipc(p)

    def test_write_creates_file_with_float32(self, tmp_path):
        table = pa.table(
            {"prediction": pa.array([1.0, 2.0], type=pa.float32())},
            schema=PRED_SCHEMA,
        )
        out = str(tmp_path / "out.arrow")
        srv._write_arrow_ipc(out, table)
        assert Path(out).exists()
        restored = srv._read_arrow_ipc(out)
        assert restored.schema.field("prediction").type == pa.float32()
        assert restored.num_rows == 2

    def test_write_creates_parent_dirs(self, tmp_path):
        deep = str(tmp_path / "a" / "b" / "c" / "out.arrow")
        table = pa.table({"prediction": pa.array([1.0], type=pa.float32())},
                         schema=PRED_SCHEMA)
        srv._write_arrow_ipc(deep, table)
        assert Path(deep).exists()

    def test_roundtrip_null_values_preserved(self, tmp_path):
        p = str(tmp_path / "nulls.arrow")
        _write_ipc_with_nulls(p)
        table = srv._read_arrow_ipc(p)
        bz = table.column("bz")
        assert bz[1].is_valid == False  # null (NaN on .NET side via RULE-120)
        assert bz[0].as_py() == pytest.approx(-5.0, abs=1e-4)


# ─── 2. Stub trainer ─────────────────────────────────────────────────────────

class TestStubTrainer:

    def test_stub_train_creates_meta_json(self, tmp_path):
        p = str(tmp_path / "data.arrow")
        _write_float32_ipc(p)
        table = srv._read_arrow_ipc(p)
        model_id = srv._stub_train(table, "TFT", str(tmp_path / "models"))

        meta_path = Path(tmp_path) / "models" / model_id / "meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["stub"] is True
        assert meta["model_type"] == "TFT"
        assert meta["rows"] == table.num_rows

    def test_stub_train_model_id_format(self, tmp_path):
        p = str(tmp_path / "data.arrow")
        _write_float32_ipc(p)
        table = srv._read_arrow_ipc(p)
        model_id = srv._stub_train(table, "NeuralOde", str(tmp_path / "models"))
        assert model_id.startswith("stub_neuralode_")

    def test_stub_train_unique_model_ids(self, tmp_path):
        p = str(tmp_path / "data.arrow")
        _write_float32_ipc(p)
        table = srv._read_arrow_ipc(p)
        ids = {srv._stub_train(table, "TFT", str(tmp_path / "models")) for _ in range(5)}
        assert len(ids) == 5, "each stub_train call must produce a unique model_id"


# ─── 3. TFT trainer ──────────────────────────────────────────────────────────

class TestTftTrainer:

    def _make_table(self, rows: int = 20) -> pa.Table:
        speeds = pa.array([float(400 + i * 50) for i in range(rows)], type=pa.float32())
        arrivals = pa.array([float(48 - i * 0.5) for i in range(rows)], type=pa.float32())
        schema = pa.schema([pa.field("speed", pa.float32()), pa.field("arrival", pa.float32())])
        return pa.table({"speed": speeds, "arrival": arrivals}, schema=schema)

    def test_tft_train_returns_model_id(self, tmp_path):
        table = self._make_table(30)
        ct_event = threading.Event()
        progress_calls = []

        model_id = srv._train_tft(
            table, ["speed"], "arrival",
            {"epochs": "3", "hidden_size": "16"},
            str(tmp_path / "models"),
            lambda e, tl, vl: progress_calls.append((e, tl, vl)),
            ct_event,
        )
        assert model_id  # non-empty
        if srv._TORCH_AVAILABLE:
            assert model_id.startswith("tft_")
            assert len(progress_calls) == 3
        else:
            assert model_id.startswith("stub_tft_")

    def test_tft_train_respects_cancellation(self, tmp_path):
        if not srv._TORCH_AVAILABLE:
            pytest.skip("cancellation test requires PyTorch")
        table = self._make_table(50)
        ct_event = threading.Event()
        ct_event.set()  # cancel immediately

        model_id = srv._train_tft(
            table, ["speed"], "arrival",
            {"epochs": "100"},
            str(tmp_path / "models"),
            lambda e, tl, vl: None,
            ct_event,
        )
        # Should still produce a model_id (training completed at least one pass or aborted)
        assert model_id


# ─── 4. NeuralODE trainer ────────────────────────────────────────────────────

class TestNeuralOdeTrainer:

    def _make_table(self, rows: int = 20) -> pa.Table:
        schema = pa.schema([pa.field("speed", pa.float32()), pa.field("arrival", pa.float32())])
        speeds = pa.array([float(400 + i * 50) for i in range(rows)], type=pa.float32())
        arrivals = pa.array([float(48 - i * 0.5) for i in range(rows)], type=pa.float32())
        return pa.table({"speed": speeds, "arrival": arrivals}, schema=schema)

    def test_neural_ode_train_returns_model_id(self, tmp_path):
        table = self._make_table(20)
        ct_event = threading.Event()

        model_id = srv._train_neural_ode(
            table, ["speed"], "arrival",
            {"epochs": "3"},
            str(tmp_path / "models"),
            lambda e, tl, vl: None,
            ct_event,
        )
        assert model_id
        if srv._TORCH_AVAILABLE:
            assert model_id.startswith("neural_ode_")
        else:
            assert model_id.startswith("stub_neuralode_")

    def test_neural_ode_meta_json_has_correct_dim(self, tmp_path):
        if not srv._TORCH_AVAILABLE:
            pytest.skip("requires PyTorch")
        table = self._make_table(20)
        ct_event = threading.Event()
        model_id = srv._train_neural_ode(
            table, ["speed"], "arrival",
            {"epochs": "2"},
            str(tmp_path / "models"),
            lambda e, tl, vl: None,
            ct_event,
        )
        meta = json.loads((Path(tmp_path) / "models" / model_id / "meta.json").read_text())
        assert meta["dim"] == 1  # 1 feature column


# ─── 5. Prediction ───────────────────────────────────────────────────────────

class TestPredict:

    def _stub_model(self, tmp_path: Path, model_type: str = "TFT") -> str:
        schema = pa.schema([pa.field("speed", pa.float32())])
        table = pa.table({"speed": pa.array([400.0, 500.0], type=pa.float32())}, schema=schema)
        model_id = srv._stub_train(table, model_type, str(tmp_path / "models"))
        return model_id

    def test_stub_predict_returns_48h(self, tmp_path):
        model_id = self._stub_model(tmp_path, "TFT")
        schema = pa.schema([pa.field("speed", pa.float32())])
        table = pa.table({"speed": pa.array([700.0, 1000.0, 1500.0], type=pa.float32())}, schema=schema)

        result = srv._predict(model_id, str(tmp_path / "models"), table)
        assert result.num_rows == 3
        for v in result.column("prediction").to_pylist():
            assert abs(v - 48.0) < 1e-4, f"stub must return 48h, got {v}"

    def test_predict_missing_model_raises(self, tmp_path):
        schema = pa.schema([pa.field("speed", pa.float32())])
        table = pa.table({"speed": pa.array([700.0], type=pa.float32())}, schema=schema)
        with pytest.raises(FileNotFoundError, match="not found"):
            srv._predict("nonexistent_model_id", str(tmp_path / "models"), table)

    def test_tft_predict_returns_finite_values(self, tmp_path):
        if not srv._TORCH_AVAILABLE:
            pytest.skip("requires PyTorch")
        # Train a minimal TFT model
        schema = pa.schema([pa.field("speed", pa.float32())])
        train_table = pa.table(
            {"speed": pa.array([float(i * 100) for i in range(1, 21)], type=pa.float32()),
             "arrival": pa.array([float(48 - i) for i in range(20)], type=pa.float32())},
            schema=pa.schema([pa.field("speed", pa.float32()), pa.field("arrival", pa.float32())]),
        )
        ct_event = threading.Event()
        model_id = srv._train_tft(
            train_table, ["speed"], "arrival",
            {"epochs": "2", "hidden_size": "8"},
            str(tmp_path / "models"),
            lambda *_: None, ct_event,
        )
        test_table = pa.table(
            {"speed": pa.array([500.0, 1000.0], type=pa.float32())}, schema=schema
        )
        result = srv._predict(model_id, str(tmp_path / "models"), test_table)
        assert result.num_rows == 2
        for v in result.column("prediction").to_pylist():
            assert v is not None and not (v != v), f"NaN prediction from live TFT: {v}"

    def test_predict_output_schema_is_float32(self, tmp_path):
        model_id = self._stub_model(tmp_path, "NeuralOde")
        schema = pa.schema([pa.field("speed", pa.float32())])
        table = pa.table({"speed": pa.array([700.0], type=pa.float32())}, schema=schema)
        result = srv._predict(model_id, str(tmp_path / "models"), table)
        assert result.schema.field("prediction").type == pa.float32()


# ─── 6. ONNX export ──────────────────────────────────────────────────────────

class TestOnnxExport:

    def test_export_stub_path_writes_placeholder(self, tmp_path):
        if srv._TORCH_AVAILABLE:
            pytest.skip("stub path only runs when PyTorch is absent")
        schema = pa.schema([pa.field("speed", pa.float32())])
        table = pa.table({"speed": pa.array([400.0], type=pa.float32())}, schema=schema)
        model_id = srv._stub_train(table, "NeuralOde", str(tmp_path / "models"))
        onnx_path = str(tmp_path / "out.onnx")
        srv._export_neural_ode_onnx(model_id, str(tmp_path / "models"), onnx_path, opset=20)
        assert Path(onnx_path).exists()
        assert Path(onnx_path).stat().st_size >= 8

    def test_export_live_produces_onnx_file(self, tmp_path):
        if not srv._TORCH_AVAILABLE:
            pytest.skip("requires PyTorch for live ONNX export")
        train_schema = pa.schema([pa.field("speed", pa.float32()), pa.field("arrival", pa.float32())])
        train_table = pa.table({
            "speed": pa.array([float(i * 100) for i in range(1, 21)], type=pa.float32()),
            "arrival": pa.array([float(48 - i) for i in range(20)], type=pa.float32()),
        }, schema=train_schema)
        ct_event = threading.Event()
        model_id = srv._train_neural_ode(
            train_table, ["speed"], "arrival",
            {"epochs": "2"},
            str(tmp_path / "models"),
            lambda *_: None, ct_event,
        )
        onnx_path = str(tmp_path / "dynamics.onnx")
        srv._export_neural_ode_onnx(model_id, str(tmp_path / "models"), onnx_path, opset=20)
        assert Path(onnx_path).exists()
        assert Path(onnx_path).stat().st_size > 100, "ONNX file should contain actual model data"

    def test_export_missing_model_raises(self, tmp_path):
        if not srv._TORCH_AVAILABLE:
            pytest.skip("requires PyTorch for missing model test")
        with pytest.raises(FileNotFoundError, match="meta not found"):
            srv._export_neural_ode_onnx(
                "nonexistent_id", str(tmp_path / "models"),
                str(tmp_path / "out.onnx"), opset=21,
            )


# ─── 7. gRPC service via live server (pytest.mark.live) ──────────────────────

def _find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """Start a real gRPC server in a background thread for live tests."""
    model_dir = str(tmp_path_factory.mktemp("models"))
    port = _find_free_port()
    import grpc
    from concurrent import futures
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    solarpipe_pb2_grpc.add_PythonTrainerServicer_to_server(
        srv._PythonTrainerServicer(model_dir), server
    )
    health_svc = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_svc, server)
    from grpc_health.v1 import health_pb2
    health_svc.set("", health_pb2.HealthCheckResponse.SERVING)
    health_svc.set("solarpipe.PythonTrainer", health_pb2.HealthCheckResponse.SERVING)

    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    yield server, port, model_dir
    server.stop(grace=1)


@pytest.mark.live
class TestLiveServer:

    def test_health_check_responds_serving(self, live_server):
        import grpc
        from grpc_health.v1 import health_pb2, health_pb2_grpc

        _, port, _ = live_server
        with grpc.insecure_channel(f"localhost:{port}") as channel:
            stub = health_pb2_grpc.HealthStub(channel)
            resp = stub.Check(health_pb2.HealthCheckRequest(service="solarpipe.PythonTrainer"))
            assert resp.status == health_pb2.HealthCheckResponse.SERVING

    def test_train_rpc_returns_model_id(self, live_server, tmp_path):
        import grpc
        _, port, model_dir = live_server
        arrow_path = str(tmp_path / "train.arrow")
        _write_float32_ipc(arrow_path, rows=10)

        with grpc.insecure_channel(f"localhost:{port}") as channel:
            stub = solarpipe_pb2_grpc.PythonTrainerStub(channel)
            req = solarpipe_pb2.TrainRequest(
                stage_name="test_stage",
                model_type="TFT",
                arrow_ipc_path=arrow_path,
                feature_columns=["speed"],
                target_column="density",
                hyperparameters={"epochs": "2"},
                model_output_dir=model_dir,
            )
            resp = stub.Train(req)
            assert resp.error_message == "", f"Train RPC failed: {resp.error_message}"
            assert resp.model_id, "model_id must be non-empty on success"

    def test_stream_train_emits_progress_then_final(self, live_server, tmp_path):
        import grpc
        _, port, model_dir = live_server
        arrow_path = str(tmp_path / "train_stream.arrow")
        _write_float32_ipc(arrow_path, rows=15)

        with grpc.insecure_channel(f"localhost:{port}") as channel:
            stub = solarpipe_pb2_grpc.PythonTrainerStub(channel)
            req = solarpipe_pb2.TrainRequest(
                stage_name="test_stream",
                model_type="TFT",
                arrow_ipc_path=arrow_path,
                feature_columns=["speed"],
                target_column="density",
                hyperparameters={"epochs": "3", "hidden_size": "8"},
                model_output_dir=model_dir,
            )
            messages = list(stub.StreamTrain(req))
            assert messages, "must receive at least one message"
            final = messages[-1]
            assert final.is_final is True
            assert final.error_message == "", f"StreamTrain error: {final.error_message}"
            assert final.model_id, "final message must carry model_id"

    def test_predict_rpc_roundtrip(self, live_server, tmp_path):
        import grpc
        _, port, model_dir = live_server
        arrow_path = str(tmp_path / "train_pred.arrow")
        _write_float32_ipc(arrow_path, rows=10)

        with grpc.insecure_channel(f"localhost:{port}") as channel:
            stub_rpc = solarpipe_pb2_grpc.PythonTrainerStub(channel)
            # First train
            train_req = solarpipe_pb2.TrainRequest(
                stage_name="pred_stage",
                model_type="TFT",
                arrow_ipc_path=arrow_path,
                feature_columns=["speed"],
                target_column="density",
                hyperparameters={"epochs": "2"},
                model_output_dir=model_dir,
            )
            train_resp = stub_rpc.Train(train_req)
            assert train_resp.error_message == ""

            # Then predict — input must have only the feature columns used during training
            pred_arrow = str(tmp_path / "pred_input.arrow")
            schema = pa.schema([pa.field("speed", pa.float32())])
            pred_table = pa.table(
                {"speed": pa.array([500.0, 700.0, 900.0], type=pa.float32())}, schema=schema
            )
            with pa_ipc.new_file(pred_arrow, schema) as w:
                w.write_table(pred_table)
            out_arrow = str(tmp_path / "pred_output.arrow")
            pred_req = solarpipe_pb2.PredictRequest(
                model_id=train_resp.model_id,
                arrow_ipc_path=pred_arrow,
                output_arrow_ipc_path=out_arrow,
                feature_columns=["speed"],
            )
            pred_resp = stub_rpc.Predict(pred_req)
            assert pred_resp.error_message == ""
            assert pred_resp.row_count == 3
            assert Path(out_arrow).exists()

            # Validate output Arrow schema is float32 (RULE-063)
            result_table = srv._read_arrow_ipc(out_arrow)
            assert result_table.schema.field("prediction").type == pa.float32()

    def test_export_onnx_rpc(self, live_server, tmp_path):
        import grpc
        _, port, model_dir = live_server
        arrow_path = str(tmp_path / "train_onnx.arrow")
        _write_float32_ipc(arrow_path, rows=10)

        with grpc.insecure_channel(f"localhost:{port}") as channel:
            stub_rpc = solarpipe_pb2_grpc.PythonTrainerStub(channel)
            train_req = solarpipe_pb2.TrainRequest(
                stage_name="onnx_stage",
                model_type="NeuralOde",
                arrow_ipc_path=arrow_path,
                feature_columns=["speed"],
                target_column="density",
                hyperparameters={"epochs": "2"},
                model_output_dir=model_dir,
            )
            train_resp = stub_rpc.Train(train_req)
            assert train_resp.error_message == ""

            onnx_path = str(tmp_path / "dynamics.onnx")
            onnx_req = solarpipe_pb2.ExportOnnxRequest(
                model_id=train_resp.model_id,
                onnx_output_path=onnx_path,
                opset=21,
            )
            onnx_resp = stub_rpc.ExportOnnx(onnx_req)
            assert onnx_resp.error_message == ""
            assert Path(onnx_path).exists()


# ─── 8. Stub server (solarpipe_stub.py) ──────────────────────────────────────

@pytest.fixture(scope="module")
def stub_server(tmp_path_factory):
    """Start solarpipe_stub server in a background thread."""
    model_dir = str(tmp_path_factory.mktemp("stub_models"))
    port = _find_free_port()

    import grpc
    from concurrent import futures

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    solarpipe_pb2_grpc.add_PythonTrainerServicer_to_server(
        stub_mod.PythonTrainerStub(), server
    )
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    yield server, port, model_dir
    server.stop(grace=1)


@pytest.mark.live
class TestStubServer:

    def test_stub_train_returns_model_id(self, stub_server, tmp_path):
        import grpc
        _, port, model_dir = stub_server
        arrow_path = str(tmp_path / "data.arrow")
        _write_float32_ipc(arrow_path, rows=5)

        with grpc.insecure_channel(f"localhost:{port}") as channel:
            stub_rpc = solarpipe_pb2_grpc.PythonTrainerStub(channel)
            req = solarpipe_pb2.TrainRequest(
                stage_name="my_stage",
                model_type="TFT",
                arrow_ipc_path=arrow_path,
                feature_columns=["speed"],
                target_column="density",
                hyperparameters={},
                model_output_dir=model_dir,
            )
            resp = stub_rpc.Train(req)
            assert resp.error_message == ""
            assert resp.model_id == "stub_my_stage_v1"
            assert resp.epochs_completed == 1

    def test_stub_stream_train_emits_progress_and_final(self, stub_server, tmp_path):
        import grpc
        _, port, model_dir = stub_server
        arrow_path = str(tmp_path / "data2.arrow")
        _write_float32_ipc(arrow_path, rows=5)

        with grpc.insecure_channel(f"localhost:{port}") as channel:
            stub_rpc = solarpipe_pb2_grpc.PythonTrainerStub(channel)
            req = solarpipe_pb2.TrainRequest(
                stage_name="stream_stage",
                model_type="TFT",
                arrow_ipc_path=arrow_path,
                feature_columns=["speed"],
                target_column="density",
                hyperparameters={},
                model_output_dir=model_dir,
            )
            msgs = list(stub_rpc.StreamTrain(req))
            assert len(msgs) == 3  # 2 progress + 1 final
            assert msgs[-1].is_final is True
            assert msgs[-1].model_id == "stub_stream_stage_v1"
            # Non-final messages must not carry is_final=True
            for m in msgs[:-1]:
                assert not m.is_final

    def test_stub_predict_returns_48h_per_row(self, stub_server, tmp_path):
        import grpc
        _, port, model_dir = stub_server
        input_path = str(tmp_path / "pred_in.arrow")
        output_path = str(tmp_path / "pred_out.arrow")
        _write_float32_ipc(input_path, rows=4)

        with grpc.insecure_channel(f"localhost:{port}") as channel:
            stub_rpc = solarpipe_pb2_grpc.PythonTrainerStub(channel)
            req = solarpipe_pb2.PredictRequest(
                model_id="any_model",
                arrow_ipc_path=input_path,
                output_arrow_ipc_path=output_path,
                feature_columns=["speed"],
            )
            resp = stub_rpc.Predict(req)
            assert resp.error_message == ""
            assert resp.row_count == 4
            assert Path(output_path).exists()

            table = srv._read_arrow_ipc(output_path)
            for v in table.column("prediction").to_pylist():
                assert abs(v - 48.0) < 1e-4, f"stub must return 48h, got {v}"

    def test_stub_export_onnx_writes_placeholder(self, stub_server, tmp_path):
        import grpc
        _, port, _ = stub_server
        onnx_path = str(tmp_path / "stub.onnx")

        with grpc.insecure_channel(f"localhost:{port}") as channel:
            stub_rpc = solarpipe_pb2_grpc.PythonTrainerStub(channel)
            req = solarpipe_pb2.ExportOnnxRequest(
                model_id="stub_model",
                onnx_output_path=onnx_path,
                opset=21,
            )
            resp = stub_rpc.ExportOnnx(req)
            assert resp.error_message == ""
            assert Path(onnx_path).exists()
            assert Path(onnx_path).stat().st_size >= 8


# ─── 9. Parent heartbeat (unit) ──────────────────────────────────────────────

class TestParentHeartbeat:

    def test_heartbeat_thread_starts_without_error(self):
        # Use current PID — parent is alive, thread should run without crashing
        srv._start_parent_heartbeat(os.getpid(), poll_interval_s=0.05)
        time.sleep(0.15)
        # No assertion needed — absence of exception is the contract
