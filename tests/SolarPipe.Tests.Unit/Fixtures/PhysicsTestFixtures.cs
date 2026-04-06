using SolarPipe.Core.Domain;

namespace SolarPipe.Tests.Unit.Fixtures;

public static class PhysicsTestFixtures
{
    // CME event 1: 2003-10-28 "Halloween Storm" — X17 flare
    // Published in Gopalswamy et al. 2005, JGRA 110, A09S15
    public static class HalloweenStorm2003
    {
        public static readonly SkyPlaneSpeed SkyPlaneSpeed = new(2459f);       // km/s CDAW catalog
        public static readonly RadialSpeed RadialSpeed = new(2459f);            // km/s (Earth-directed, correction ~1.0)
        public const float ProtonDensityPerCm3 = 42.0f;
        public const float AmbientSolarWindKmPerSec = 530f;
        public const float GsmBzNt = -35.0f;                                   // GSM-frame Bz (nT)
        public const float DstMinNt = -383.0f;                                 // Published Dst minimum
        public static readonly DateTime LaunchTime = new(2003, 10, 28, 11, 30, 0, DateTimeKind.Utc);
        public static readonly DateTime ArrivalTime = new(2003, 10, 29, 6, 12, 0, DateTimeKind.Utc);
        public static readonly TimeSpan TransitTime = ArrivalTime - LaunchTime;
    }

    // CME event 2: 2012-07-23 — near-Carrington event (missed Earth)
    // Published in Baker et al. 2013, Space Weather 11, 585
    public static class July2012Event
    {
        public static readonly SkyPlaneSpeed SkyPlaneSpeed = new(2003f);
        public static readonly RadialSpeed RadialSpeed = new(2003f);
        public const float ProtonDensityPerCm3 = 79.0f;
        public const float AmbientSolarWindKmPerSec = 490f;
        public const float GsmBzNt = -52.0f;                                   // Estimated GSM Bz
        public static readonly DateTime LaunchTime = new(2012, 7, 23, 2, 15, 0, DateTimeKind.Utc);
    }

    // CME event 3: Moderate event for numerical stability testing
    public static class ModerateEvent
    {
        public static readonly SkyPlaneSpeed SkyPlaneSpeed = new(800f);
        public static readonly RadialSpeed RadialSpeed = new(800f);
        public const float ProtonDensityPerCm3 = 10.0f;
        public const float AmbientSolarWindKmPerSec = 400f;
        public const float GsmBzNt = -10.0f;
        public const float DstExpectedNt = -80.0f;                             // Approximate expected
        public static readonly DateTime LaunchTime = new(2015, 6, 18, 8, 0, 0, DateTimeKind.Utc);
    }

    // Drag model parameters — from Vršnak et al. 2013 (A&A 512)
    public static class DragParameters
    {
        // Drag coefficient (CD) × cross-section / mass — typical CME range
        public const float GammaMinKmInv = 0.2e-7f;    // Low-drag: massive CME
        public const float GammaTypicalKmInv = 0.5e-7f; // Typical
        public const float GammaMaxKmInv = 2.0e-7f;    // High-drag: small CME

        // Initial heliocentric distances for ODE integration (in solar radii)
        public const float StartDistanceSolarRadii = 21.5f;   // ENLIL inner boundary
        public const float EarthDistanceSolarRadii = 215.0f;  // 1 AU in solar radii
    }

    // Solar wind nominal conditions (O'Brien & McPherron 2000 reference)
    public static class NominalSolarWind
    {
        public const float SpeedKmPerSec = 400f;
        public const float DensityPerCm3 = 7.0f;
        public const float BzGsmNt = -2.0f;
        public const float TemperatureK = 1.2e5f;
    }

    // Coupling function test values — Newell et al. 2007
    public static class NewellCouplingValues
    {
        // Input: v=450 km/s, Bt=10 nT, clock_angle=180° (purely southward)
        public const float SolarWindSpeedKmPerSec = 450f;
        public const float BtNt = 10.0f;
        public const float ClockAngleDeg = 180f;
        public const float ExpectedCouplingMwb = 2.5e3f;   // ~2500 Wb/s (approximate)
    }

    // OMNI data sample timestamps for parser tests
    public static class OmniTimestamps
    {
        public const string ValidOmni = "2003 301 1130";     // 2003-10-28 11:30 UTC
        public const string ValidOmniMidnight = "2003 001 0000"; // 2003-01-01 00:00 UTC
        public const string InvalidOmni = "2003 400 9999";   // Invalid DOY
    }

    // CDAW catalog timestamps
    public static class CdawTimestamps
    {
        public const string ValidSlash = "2003/10/28 11:30";
        public const string ValidIso = "2003-10-28T11:30:00";
        public const string ValidIsoZ = "2003-10-28T11:30:00Z";
    }

    // Sentinel values used in OMNI/ACE data
    public const float OmniSentinel = 9999.9f;
    public const float AceSentinel = -1e31f;
    public const float OmniSentinelInt = 999f;
}
