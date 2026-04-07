using System.Net;
using System.Text;
using FluentAssertions;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Data.Providers;
using SolarPipe.Prediction;
using SolarPipe.Training.Adapters;
using SolarPipe.Training.Physics;

namespace SolarPipe.Tests.Integration;

// Phase 4 end-to-end integration tests.
//
// Verifies all four framework adapters are operational and the Phase 4 physics
// models integrate correctly with the composition algebra.
//
// These tests do NOT require a live gRPC server or GPU — they test contracts,
// data flows, and the physics computation paths with real adapters.
[Trait("Category", "Integration")]
public sealed class Phase4EndToEndTests : IDisposable
{
    private readonly string _tempDir;

    public Phase4EndToEndTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"phase4_e2e_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); } catch { /* best-effort */ }
    }

    // ─── 1. All four framework adapters have correct FrameworkType ─────────────────

    [Fact]
    public void AllFourAdapters_CorrectFrameworkTypes()
    {
        // ML.NET
        IFrameworkAdapter mlNet = new MlNetAdapter();
        mlNet.FrameworkType.Should().Be(FrameworkType.MlNet);

        // Physics (includes BurtonOde + NewellCoupling from Phase 4)
        IFrameworkAdapter physics = new PhysicsAdapter();
        physics.FrameworkType.Should().Be(FrameworkType.Physics);
        physics.SupportedModels.Should().Contain("BurtonOde");
        physics.SupportedModels.Should().Contain("NewellCoupling");

        // ONNX
        IFrameworkAdapter onnx = new OnnxAdapter();
        onnx.FrameworkType.Should().Be(FrameworkType.Onnx);
        onnx.SupportedModels.Should().Contain("Standard");
        onnx.SupportedModels.Should().Contain("NeuralOde");

        // Python gRPC
        using var grpc = new GrpcSidecarAdapter("http://localhost:59999", _tempDir);
        ((IFrameworkAdapter)grpc).FrameworkType.Should().Be(FrameworkType.PythonGrpc);
        ((IFrameworkAdapter)grpc).SupportedModels.Should().Contain("TFT");
        ((IFrameworkAdapter)grpc).SupportedModels.Should().Contain("NeuralOde");
    }

    // ─── 2. BurtonOde + NewellCoupling composition pipeline ──────────────────────
    // Realistic scenario: NewellCoupling computes reconnection rate from solar wind,
    // BurtonOde predicts Dst minimum — sequential physics pipeline.

    [Fact]
    public async Task Phase4_BurtonOdeAndNewellCoupling_SequentialPhysicsPipeline()
    {
        var adapter = new PhysicsAdapter();

        // Solar wind data: 3 intervals (Halloween storm conditions)
        float[] byGsm  = [2.0f,  5.0f, -3.0f];
        float[] bzGsm  = [-35.0f, -20.0f, -10.0f]; // GSM-frame southward
        float[] vKmS   = [530.0f, 450.0f, 400.0f];
        float[] bzBurt = [-35.0f, -20.0f, -10.0f]; // same data for Burton

        var schema = new DataSchema([
            new ColumnInfo("by_gsm", ColumnType.Float, false),
            new ColumnInfo("bz_gsm", ColumnType.Float, false),
            new ColumnInfo("v_km_s", ColumnType.Float, false),
        ]);
        using var solarWindFrame = new InMemoryDataFrame(schema, [byGsm, bzGsm, vKmS]);

        // Stage 1: Newell coupling
        var newellConfig = new StageConfig(
            "newell_stage", "Physics", "NewellCoupling", "src",
            ["by_gsm", "bz_gsm", "v_km_s"], "coupling");
        var newellModel = await adapter.TrainAsync(newellConfig, solarWindFrame, null, CancellationToken.None);
        var couplingResult = await newellModel.PredictAsync(solarWindFrame, CancellationToken.None);

        // Southward, high-speed intervals → positive coupling
        couplingResult.Values.Should().HaveCount(3);
        couplingResult.Values[0].Should().BeGreaterThan(0f, "southward Bz at 530 km/s → positive coupling");
        couplingResult.Values[1].Should().BeGreaterThan(0f, "southward Bz at 450 km/s → positive coupling");

        // Coupling should decrease with speed (v[0]=530 > v[1]=450 → coupling[0] > coupling[1])
        couplingResult.Values[0].Should().BeGreaterThan(couplingResult.Values[1],
            "higher speed must produce larger coupling (v^4/3 scaling)");

        // Stage 2: BurtonOde (each row represents a 1-hour interval with fixed VBs)
        var burtonConfig = new StageConfig(
            "burton_stage", "Physics", "BurtonOde", "src",
            ["bz_gsm", "v_km_s"], "dst_min",
            new Dictionary<string, object>
            {
                ["dst0_nt"] = 0.0,
                ["dt_hours"] = 1.0,
                ["pdyn_npa"] = 5.0
            });
        var burtonModel = await adapter.TrainAsync(burtonConfig, solarWindFrame, null, CancellationToken.None);
        var dstResult = await burtonModel.PredictAsync(solarWindFrame, CancellationToken.None);

        dstResult.Values.Should().HaveCount(3);
        // All southward Bz rows should produce negative Dst
        dstResult.Values[0].Should().BeLessThan(0f, "Halloween storm southward Bz drives negative Dst");
        dstResult.Values[1].Should().BeLessThan(0f, "moderate southward Bz drives negative Dst");
        // Stronger southward (row 0, Bz=-35) should be more negative than weaker (row 2, Bz=-10)
        dstResult.Values[0].Should().BeLessThan(dstResult.Values[2],
            "stronger southward Bz must produce more negative Dst");
    }

    // ─── 3. RestApiProvider schema discovery with SWPC JSON ──────────────────────

    [Fact]
    public async Task Phase4_RestApiProvider_SwpcSolarWind_DiscoverSchema()
    {
        const string sampleJson = """
            [
              {"time_tag":"2024-01-01 00:00:00","bx_gsm":-2.1,"by_gsm":3.4,"bz_gsm":-5.1,
               "speed":425.0,"density":5.2,"temperature":80000},
              {"time_tag":"2024-01-01 00:01:00","bx_gsm":-1.8,"by_gsm":2.9,"bz_gsm":-4.7,
               "speed":428.0,"density":5.0,"temperature":78000}
            ]
            """;

        var handler = new StubHttpHandler(sampleJson);
        var http = new HttpClient(handler);
        var provider = new RestApiProvider(http);

        var config = new DataSourceConfig(
            "swpc_test", "rest_api",
            "https://stub.test/rtsw_wind_1m.json",
            new Dictionary<string, string> { ["endpoint_type"] = "swpc_solar_wind" });

        // Schema discovery
        var schema = await provider.DiscoverSchemaAsync(config, CancellationToken.None);
        schema.Columns.Should().HaveCountGreaterOrEqualTo(5,
            "SWPC solar wind schema must include bz_gsm, speed, density, temperature, and others");
        schema.HasColumn("bz_gsm").Should().BeTrue();
        schema.HasColumn("speed").Should().BeTrue();

        // LoadAsync
        using var frame = await provider.LoadAsync(config, new DataQuery(), CancellationToken.None);
        frame.RowCount.Should().Be(2);
        float[] bz = frame.GetColumn("bz_gsm");
        bz[0].Should().BeApproximately(-5.1f, 1e-3f);
    }

    // ─── 4. OnnxAdapter contract: FrameworkType, SupportedModels, error paths ────

    [Fact]
    public async Task Phase4_OnnxAdapter_Contract_VerifiesAllPaths()
    {
        var adapter = new OnnxAdapter();

        // FrameworkType
        adapter.FrameworkType.Should().Be(FrameworkType.Onnx);

        // SupportedModels
        adapter.SupportedModels.Should().Contain("Standard");
        adapter.SupportedModels.Should().Contain("NeuralOde");

        // Missing model_path → ArgumentException
        var missingPathConfig = new StageConfig(
            "onnx_stage", "Onnx", "Standard", "src",
            ["feature_a"], "target",
            new Dictionary<string, object>());

        Func<Task> missingPath = () => adapter.TrainAsync(
            missingPathConfig, null!, null, CancellationToken.None);
        await missingPath.Should().ThrowAsync<ArgumentException>().WithMessage("*model_path*");

        // Nonexistent file → FileNotFoundException
        var noFileConfig = new StageConfig(
            "onnx_stage", "Onnx", "Standard", "src",
            ["feature_a"], "target",
            new Dictionary<string, object>
            {
                ["model_path"] = Path.Combine(_tempDir, "nonexistent_phase4.onnx")
            });

        Func<Task> noFile = () => adapter.TrainAsync(
            noFileConfig, null!, null, CancellationToken.None);
        await noFile.Should().ThrowAsync<FileNotFoundException>().WithMessage("*ONNX model file not found*");
    }

    // ─── helpers ─────────────────────────────────────────────────────────────────

    // Stub HttpMessageHandler for RestApiProvider tests (no live network).
    private sealed class StubHttpHandler(string json) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var response = new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(json, Encoding.UTF8, "application/json")
            };
            return Task.FromResult(response);
        }
    }
}
