using FluentAssertions;
using Microsoft.Data.Sqlite;
using SolarPipe.Data;
using SolarPipe.Training.Physics;

namespace SolarPipe.Tests.Unit;

// Phase 9 §6.1: DensityModulatedDrag unit tests.
//
// γ_eff(t) = γ₀ · (n_obs(t) / n_ref), clipped to [1e-9, 1e-6] km⁻¹.
// Missing n_obs → fall back to γ₀ with FellBack=true.
//
// These tests spin up an in-memory SQLite DB with a tiny omni_hourly table
// per case so L1ObservationStream.LoadFromSqlite can be exercised end-to-end.
[Trait("Category", "Unit")]
public sealed class DensityModulatedDragTests
{
    private const double Gamma0 = 0.5e-7; // typical drag parameter
    private const double NRef   = 5.0;    // quiet-sun reference

    private static readonly DateTime LaunchTime = new(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);

    // Build an in-memory SQLite DB with omni_hourly populated with the given
    // (flow_speed, proton_density) sequence starting at LaunchTime. Returns a
    // connection string that stays valid while the returned connection is alive.
    private static (SqliteConnection Conn, string ConnString) CreateOmniDb(
        (double? V, double? N)[] hours)
    {
        // Use a named shared in-memory DB so LoadFromSqlite can open its own
        // connection against the same store.
        var name = $"omnitest_{Guid.NewGuid():N}";
        var connString = $"Data Source={name};Mode=Memory;Cache=Shared";
        var keepAlive = new SqliteConnection(connString);
        keepAlive.Open();

        using var cmd = keepAlive.CreateCommand();
        cmd.CommandText =
            @"CREATE TABLE omni_hourly (
                datetime       TEXT PRIMARY KEY,
                flow_speed     REAL,
                proton_density REAL,
                Bz_GSM         REAL);";
        cmd.ExecuteNonQuery();

        for (int i = 0; i < hours.Length; i++)
        {
            var ts = LaunchTime.AddHours(i).ToString("yyyy-MM-dd HH:mm");
            using var ins = keepAlive.CreateCommand();
            ins.CommandText =
                "INSERT INTO omni_hourly (datetime, flow_speed, proton_density, Bz_GSM) " +
                "VALUES (@t, @v, @n, NULL)";
            ins.Parameters.AddWithValue("@t", ts);
            ins.Parameters.AddWithValue("@v", (object?)hours[i].V ?? DBNull.Value);
            ins.Parameters.AddWithValue("@n", (object?)hours[i].N ?? DBNull.Value);
            ins.ExecuteNonQuery();
        }
        return (keepAlive, connString);
    }

    private static L1ObservationStream LoadStream((double? V, double? N)[] hours, double windowHours = 72.0)
    {
        var (conn, connString) = CreateOmniDb(hours);
        try
        {
            return L1ObservationStream.LoadFromSqlite(connString, LaunchTime, windowHours);
        }
        finally
        {
            // Stream holds loaded rows in memory — safe to close the keep-alive.
            conn.Dispose();
        }
    }

    [Fact]
    public void Gamma_AtReferenceDensity_EqualsGamma0()
    {
        var stream = LoadStream([(400.0, NRef), (400.0, NRef), (400.0, NRef)]);
        var drag = new DensityModulatedDrag(Gamma0, stream, NRef);

        var step = drag.GammaAtHour(1, speedKmPerSec: 800.0, rSolarRadii: 50.0);

        step.GammaKmInv.Should().BeApproximately(Gamma0, 1e-20);
        step.FellBack.Should().BeFalse();
        step.NObs.Should().Be(NRef);
        step.VObs.Should().Be(400.0);
    }

    [Fact]
    public void Gamma_AtDoubleDensity_EqualsTwoGamma0()
    {
        var stream = LoadStream([(400.0, NRef), (400.0, 2 * NRef), (400.0, 2 * NRef)]);
        var drag = new DensityModulatedDrag(Gamma0, stream, NRef);

        var step = drag.GammaAtHour(1, 800.0, 50.0);

        step.GammaKmInv.Should().BeApproximately(2.0 * Gamma0, 1e-20);
        step.FellBack.Should().BeFalse();
    }

    [Fact]
    public void Gamma_AtZeroDensity_ClippedToLowerBound()
    {
        var stream = LoadStream([(400.0, 5.0), (400.0, 0.0), (400.0, 0.0)]);
        var drag = new DensityModulatedDrag(Gamma0, stream, NRef);

        var step = drag.GammaAtHour(1, 800.0, 50.0);

        step.GammaKmInv.Should().Be(DensityModulatedDrag.MinGammaKmInv);
        step.FellBack.Should().BeFalse(); // value was present, just clipped
    }

    [Fact]
    public void Gamma_AtExtremeDensity_ClippedToUpperBound()
    {
        // n_obs = 1000 cm⁻³ → γ₀ · 200 = 1.0e-5, clipped to 1e-6.
        var stream = LoadStream([(400.0, 5.0), (400.0, 1000.0), (400.0, 1000.0)]);
        var drag = new DensityModulatedDrag(Gamma0, stream, NRef);

        var step = drag.GammaAtHour(1, 800.0, 50.0);

        step.GammaKmInv.Should().Be(DensityModulatedDrag.MaxGammaKmInv);
        step.FellBack.Should().BeFalse();
    }

    [Fact]
    public void Gamma_AtNullDensity_FallsBackToGamma0()
    {
        var stream = LoadStream([(400.0, NRef), (400.0, (double?)null), (400.0, NRef)]);
        var drag = new DensityModulatedDrag(Gamma0, stream, NRef);

        var step = drag.GammaAtHour(1, 800.0, 50.0);

        step.GammaKmInv.Should().Be(Gamma0);
        step.FellBack.Should().BeTrue();
        step.NObs.Should().BeNull();
        step.VObs.Should().Be(400.0);
    }

    [Fact]
    public void Gamma_OutOfWindow_FallsBackToGamma0()
    {
        var stream = LoadStream([(400.0, NRef), (400.0, NRef)]);
        var drag = new DensityModulatedDrag(Gamma0, stream, NRef);

        // Hour 50 is outside the loaded window → TryGetAtHour returns false.
        var step = drag.GammaAtHour(50, 800.0, 50.0);

        step.GammaKmInv.Should().Be(Gamma0);
        step.FellBack.Should().BeTrue();
        step.NObs.Should().BeNull();
        step.VObs.Should().BeNull();
    }

    [Fact]
    public void Ctor_RejectsGamma0OutsideBounds()
    {
        var stream = LoadStream([(400.0, NRef)]);

        Action tooLow  = () => new DensityModulatedDrag(1e-12, stream, NRef);
        Action tooHigh = () => new DensityModulatedDrag(1e-3,  stream, NRef);

        tooLow .Should().Throw<ArgumentOutOfRangeException>();
        tooHigh.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Ctor_RejectsNRefOutsideBounds()
    {
        var stream = LoadStream([(400.0, NRef)]);

        Action zero    = () => new DensityModulatedDrag(Gamma0, stream, 0.0);
        Action negative = () => new DensityModulatedDrag(Gamma0, stream, -1.0);
        Action tooHigh = () => new DensityModulatedDrag(Gamma0, stream, 200.0);

        zero.Should().Throw<ArgumentOutOfRangeException>();
        negative.Should().Throw<ArgumentOutOfRangeException>();
        tooHigh.Should().Throw<ArgumentOutOfRangeException>();
    }
}
