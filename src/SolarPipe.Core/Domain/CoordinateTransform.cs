namespace SolarPipe.Core.Domain;

// Hapgood (1992) GSE ↔ GSM coordinate transformations.
// Reference: Hapgood M.A. (1992) "Space physics coordinate transformations: A user guide",
//            Planet. Space Sci. 40(5), 711-717. https://doi.org/10.1016/0032-0633(92)90012-D
//
// GSE (Geocentric Solar Ecliptic): X toward Sun, Z = ecliptic north, Y completes right-hand.
// GSM (Geocentric Solar Magnetospheric): X toward Sun, Z = ecliptic north rotated to contain
//     the magnetic dipole axis. Related to GSE by rotation about X by the dipole tilt angle ψ.
//
// RULE-031: ALL Bz values for physics equations must be in GSM frame.
// RULE-130: Use GseVector/GsmVector typed structs — no bare float triplets.
//
// Dipole epoch: IGRF-13 (2020) — geomagnetic north pole at 80.65°N, 72.68°W (= 287.32°E).
// ψ typically varies between −34° and +34° depending on season and UT.
public static class CoordinateTransform
{
    // Geomagnetic north dipole pole (IGRF-13, 2020 epoch).
    private const double DipoleLatitudeDeg = 80.65;
    private const double DipoleLongitudeDeg = 287.32;

    // GSE → GSM rotation (about X-axis by dipole tilt angle ψ).
    // When ψ > 0: dipole north pole is tilted toward Sun (northern summer).
    //
    // Matrix: Rx(ψ)
    //   [ 1    0       0    ]
    //   [ 0  cos(ψ)  sin(ψ) ]
    //   [ 0 -sin(ψ)  cos(ψ) ]
    public static GsmVector GseToGsm(GseVector gse, DateTime utc)
    {
        double psi = DipoleTiltAngleRad(utc);
        double cosPsi = Math.Cos(psi);
        double sinPsi = Math.Sin(psi);

        float bx = gse.Bx;
        float by = (float)(cosPsi * gse.By + sinPsi * gse.Bz);
        float bz = (float)(-sinPsi * gse.By + cosPsi * gse.Bz);

        return new GsmVector(bx, by, bz);
    }

    // GSM → GSE: inverse rotation (transpose = Rx(-ψ)).
    public static GseVector GsmToGse(GsmVector gsm, DateTime utc)
    {
        double psi = DipoleTiltAngleRad(utc);
        double cosPsi = Math.Cos(psi);
        double sinPsi = Math.Sin(psi);

        float bx = gsm.Bx;
        float by = (float)(cosPsi * gsm.By - sinPsi * gsm.Bz);
        float bz = (float)(sinPsi * gsm.By + cosPsi * gsm.Bz);

        return new GseVector(bx, by, bz);
    }

    // DipoleTiltAngle: returns ψ in radians.
    // ψ > 0 when the geomagnetic north pole tilts toward the Sun (northern summer).
    //
    // Algorithm: Hapgood (1992) Appendix B.
    //   1. Compute Julian centuries T₀ from J2000.0
    //   2. Compute Sun's apparent ecliptic longitude λ_sun
    //   3. Compute Greenwich Sidereal Time θ_GST
    //   4. Express geomagnetic dipole unit vector in GSE coordinates
    //   5. ψ = atan2(D_y_gse, D_z_gse)
    //
    // The dipole unit vector in GSE is computed by the transformation chain:
    //   Geographic → GEI (via θ_GST longitude shift)
    //   GEI → GSE (via the direct 3×3 matrix using λ_sun and obliquity ε)
    public static double DipoleTiltAngleRad(DateTime utc)
    {
        double jd = DateToJulianDay(utc);

        // Step 1: Julian centuries from J2000.0
        double t0 = (jd - 2451545.0) / 36525.0;

        // Step 2: Sun's mean longitude and anomaly (degrees)
        double lambdaM = Mod360(280.46646 + 36000.76983 * t0);
        double M = ToRad(Mod360(357.52911 + 35999.05029 * t0));

        // Sun's apparent ecliptic longitude (first-order eccentricity correction)
        double lambdaSunDeg = Mod360(lambdaM + 1.914602 * Math.Sin(M) + 0.019993 * Math.Sin(2 * M));
        double lambdaSun = ToRad(lambdaSunDeg);

        // Obliquity of the ecliptic
        double epsilonDeg = 23.439291 - 0.013004 * t0;
        double epsilon = ToRad(epsilonDeg);

        // Step 3: Greenwich Sidereal Time θ_GST (Hapgood eq. B.5).
        // T₀_0h = Julian centuries from J2000.0 to 0h UT of this date (t0 already includes UT fraction).
        // To get 0h-UT centuries: subtract the UT fraction from t0.
        double utHours = utc.Hour + utc.Minute / 60.0 + utc.Second / 3600.0;
        double t0_0h = t0 - utHours / (36525.0 * 24.0);   // centuries to 0h UT
        double thetaGST = ToRad(Mod360(100.4606184 + 36000.77004 * t0_0h + 360.98564724 * (utHours / 24.0)));

        // Step 4a: Dipole unit vector in GEI.
        // The geographic longitude of the dipole pole becomes GEI longitude λ₀ + θ_GST
        // (Earth's rotation shifts geographic longitudes eastward by θ_GST relative to GEI).
        double phi0 = ToRad(DipoleLatitudeDeg);
        double lambda0 = ToRad(DipoleLongitudeDeg);
        double lambdaDip_GEI = lambda0 + thetaGST;

        double dx_gei = Math.Cos(phi0) * Math.Cos(lambdaDip_GEI);
        double dy_gei = Math.Cos(phi0) * Math.Sin(lambdaDip_GEI);
        double dz_gei = Math.Sin(phi0);

        // Step 4b: GEI → GSE using the direct transformation matrix.
        // The Sun's direction in GEI: (cos λ_sun, sin λ_sun · cos ε, sin λ_sun · sin ε).
        // The ecliptic north in GEI: (0, -sin ε, cos ε).
        // GSE Y = ecliptic north × GSE X (right-hand):
        //   Y = (0, -sin ε, cos ε) × (cos λ, sin λ cos ε, sin λ sin ε)
        //     = (-sin ε · sin λ sin ε - cos ε · sin λ cos ε,
        //         cos ε · cos λ                             - 0,
        //         0                                          - (-sin ε) · cos λ)
        //     = (-sin λ, cos λ · cos ε, cos λ · sin ε) — wait let me compute properly
        //
        // The transformation matrix M_GEI→GSE has rows = GSE basis vectors expressed in GEI:
        //   Row 0 (GSE X in GEI) = (cos λ_sun, sin λ_sun cos ε, sin λ_sun sin ε)
        //   Row 2 (GSE Z in GEI) = (0, -sin ε, cos ε)
        //   Row 1 (GSE Y in GEI) = cross product Z × X... actually easier to just apply:
        //
        // dx_gse = cos(λ)·dx_gei + sin(λ)cos(ε)·dy_gei + sin(λ)sin(ε)·dz_gei
        // dz_gse = 0·dx_gei - sin(ε)·dy_gei + cos(ε)·dz_gei

        double cosL = Math.Cos(lambdaSun), sinL = Math.Sin(lambdaSun);
        double cosE = Math.Cos(epsilon), sinE = Math.Sin(epsilon);

        double dx_gse = cosL * dx_gei + sinL * cosE * dy_gei + sinL * sinE * dz_gei;
        // GSE Y = (−sin λ, cos λ cos ε, cos λ sin ε) in GEI
        double dy_gse = -sinL * dx_gei + cosL * cosE * dy_gei + cosL * sinE * dz_gei;
        // GSE Z = (0, −sin ε, cos ε) in GEI
        double dz_gse = -sinE * dy_gei + cosE * dz_gei;

        // Step 5: dipole tilt angle ψ = atan2(D_y_gse, D_z_gse)
        return Math.Atan2(dy_gse, dz_gse);
    }

    // ─── helpers ──────────────────────────────────────────────────────────────────

    // Reduce angle to [0, 360).
    public static double Mod360(double degrees)
    {
        double v = degrees % 360.0;
        return v < 0 ? v + 360.0 : v;
    }

    public static double ToRad(double degrees) => degrees * Math.PI / 180.0;
    public static double ToDeg(double radians) => radians * 180.0 / Math.PI;

    // Compute Julian Day Number from DateTime (UTC).
    // Algorithm: Meeus "Astronomical Algorithms" chapter 7.
    public static double DateToJulianDay(DateTime utc)
    {
        int y = utc.Year;
        int m = utc.Month;
        double d = utc.Day + utc.Hour / 24.0 + utc.Minute / 1440.0 + utc.Second / 86400.0;

        if (m <= 2) { y -= 1; m += 12; }

        int a = y / 100;
        int b = 2 - a + a / 4;

        return Math.Floor(365.25 * (y + 4716)) + Math.Floor(30.6001 * (m + 1)) + d + b - 1524.5;
    }
}
