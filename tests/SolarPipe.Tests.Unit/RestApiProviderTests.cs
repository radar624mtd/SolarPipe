using System.Net;
using System.Text;
using FluentAssertions;
using SolarPipe.Core.Models;
using SolarPipe.Data.Providers;

namespace SolarPipe.Tests.Unit;

// Stub HttpMessageHandler that returns a fixed response — avoids live network calls.
file sealed class StubHttpHandler(string json, HttpStatusCode status = HttpStatusCode.OK)
    : HttpMessageHandler
{
    protected override Task<HttpResponseMessage> SendAsync(
        HttpRequestMessage request, CancellationToken cancellationToken)
    {
        var response = new HttpResponseMessage(status)
        {
            Content = new StringContent(json, Encoding.UTF8, "application/json")
        };
        return Task.FromResult(response);
    }
}

[Trait("Category", "Unit")]
public class RestApiProviderTests
{
    private static RestApiProvider MakeProvider(string json, HttpStatusCode status = HttpStatusCode.OK)
    {
        var handler = new StubHttpHandler(json, status);
        var http = new HttpClient(handler);
        return new RestApiProvider(http);
    }

    private static DataSourceConfig SwpcConfig(string? endpointType = "swpc_solar_wind") =>
        new DataSourceConfig(
            Name: "swpc_test",
            Provider: "rest_api",
            ConnectionString: "https://stub.test/rtsw_wind_1m.json",
            Options: endpointType is null ? null
                : new Dictionary<string, string> { ["endpoint_type"] = endpointType });

    [Fact]
    public void CanHandle_RestApiProvider_ReturnsTrue()
    {
        var provider = MakeProvider("[]");
        var config = SwpcConfig();
        provider.CanHandle(config).Should().BeTrue();
    }

    [Fact]
    public void CanHandle_NonRestApiProvider_ReturnsFalse()
    {
        var provider = MakeProvider("[]");
        var config = new DataSourceConfig("x", "csv", "file.csv");
        provider.CanHandle(config).Should().BeFalse();
    }

    [Fact]
    public async Task LoadAsync_SwpcSolarWind_ParsesColumnsAndNullAsNaN()
    {
        // DSCOVR safe-hold gap: null values → float.NaN (RULE-120)
        const string json = """
            [
              {"time_tag":"2024-01-01 00:00:00","bx_gsm":-2.1,"by_gsm":3.4,"bz_gsm":-5.1,"speed":425.0,"density":5.2,"temperature":80000},
              {"time_tag":"2024-01-01 00:01:00","bx_gsm":null,"by_gsm":null,"bz_gsm":null,"speed":null,"density":null,"temperature":null},
              {"time_tag":"2024-01-01 00:02:00","bx_gsm":1.5,"by_gsm":-2.0,"bz_gsm":-8.3,"speed":450.0,"density":6.1,"temperature":90000}
            ]
            """;

        var provider = MakeProvider(json);
        var config = SwpcConfig("swpc_solar_wind");
        using var frame = await provider.LoadAsync(config, new DataQuery(), CancellationToken.None);

        frame.Should().NotBeNull();
        frame.RowCount.Should().Be(3);

        // Row 0: valid data
        float[] bz = frame.GetColumn("bz_gsm");
        bz[0].Should().BeApproximately(-5.1f, 1e-4f, "first row bz_gsm should parse correctly");

        // Row 1: null → NaN (DSCOVR safe-hold)
        float.IsNaN(bz[1]).Should().BeTrue("null bz_gsm must become NaN per RULE-120");

        // Row 2: valid data
        float[] speed = frame.GetColumn("speed");
        speed[2].Should().BeApproximately(450.0f, 1e-4f);
    }

    [Fact]
    public async Task LoadAsync_DonkiCme_ParsesSpeedAndHandlesMissingAnalysis()
    {
        const string json = """
            [
              {"activityID":"2024-01-01T12:00:00-CME-001","cmeAnalyses":[{"speed":900.0,"halfAngle":35.0,"latitude":10.0,"longitude":-15.0}]},
              {"activityID":"2024-01-02T06:00:00-CME-002","cmeAnalyses":[]},
              {"activityID":"2024-01-03T09:00:00-CME-003","cmeAnalyses":[{"speed":650.0,"halfAngle":22.0,"latitude":-5.0,"longitude":30.0}]}
            ]
            """;

        var provider = MakeProvider(json);
        var config = new DataSourceConfig(
            "donki_test", "rest_api",
            "https://stub.test/CMEAnalysisList.json",
            new Dictionary<string, string> { ["endpoint_type"] = "donki_cme" });

        using var frame = await provider.LoadAsync(config, new DataQuery(), CancellationToken.None);

        frame.RowCount.Should().Be(3);
        float[] speeds = frame.GetColumn("speed");

        speeds[0].Should().BeApproximately(900.0f, 1e-3f, "first CME speed");
        float.IsNaN(speeds[1]).Should().BeTrue("CME with no analyses should produce NaN speed");
        speeds[2].Should().BeApproximately(650.0f, 1e-3f, "third CME speed");
    }

    [Fact]
    public async Task LoadAsync_WithLimit_RespectsRowCap()
    {
        const string json = """
            [
              {"bx_gsm":-1.0,"by_gsm":0.5,"bz_gsm":-2.0,"speed":400.0,"density":4.0,"temperature":70000},
              {"bx_gsm":-2.0,"by_gsm":1.0,"bz_gsm":-3.0,"speed":420.0,"density":5.0,"temperature":75000},
              {"bx_gsm":-3.0,"by_gsm":1.5,"bz_gsm":-4.0,"speed":440.0,"density":6.0,"temperature":80000},
              {"bx_gsm":-4.0,"by_gsm":2.0,"bz_gsm":-5.0,"speed":460.0,"density":7.0,"temperature":85000}
            ]
            """;

        var provider = MakeProvider(json);
        var config = SwpcConfig("swpc_solar_wind");
        using var frame = await provider.LoadAsync(config, new DataQuery(Limit: 2), CancellationToken.None);

        frame.RowCount.Should().Be(2, "Limit=2 must cap the number of rows loaded");
    }
}
