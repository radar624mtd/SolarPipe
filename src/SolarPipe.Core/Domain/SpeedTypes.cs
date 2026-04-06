namespace SolarPipe.Core.Domain;

public readonly record struct SkyPlaneSpeed(float KmPerSec)
{
    public static SkyPlaneSpeed Create(float kmPerSec)
    {
        if (kmPerSec < 0f)
            throw new ArgumentOutOfRangeException(nameof(kmPerSec), "Sky-plane speed must be non-negative.");
        return new SkyPlaneSpeed(kmPerSec);
    }
}

public readonly record struct RadialSpeed(float KmPerSec)
{
    public static RadialSpeed Create(float kmPerSec)
    {
        if (kmPerSec < 200f || kmPerSec > 3500f)
            throw new ArgumentOutOfRangeException(nameof(kmPerSec), $"Radial speed {kmPerSec} km/s outside valid range [200, 3500].");
        return new RadialSpeed(kmPerSec);
    }
}
