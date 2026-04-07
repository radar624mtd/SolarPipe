# SolarPipe .NET 8 host — Docker image.
# Multi-stage build: sdk → publish → runtime.
#
# Usage:
#   docker build -t solarpipe-host:latest .
#   docker run --rm solarpipe-host:latest --help

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build

WORKDIR /src

# Restore dependencies (layer-cached unless csproj/props files change)
COPY Directory.Packages.props         .
COPY global.json                      .
COPY SolarPipe.sln                    .
COPY src/SolarPipe.Core/SolarPipe.Core.csproj                         src/SolarPipe.Core/
COPY src/SolarPipe.Config/SolarPipe.Config.csproj                     src/SolarPipe.Config/
COPY src/SolarPipe.Data/SolarPipe.Data.csproj                         src/SolarPipe.Data/
COPY src/SolarPipe.Training/SolarPipe.Training.csproj                 src/SolarPipe.Training/
COPY src/SolarPipe.Prediction/SolarPipe.Prediction.csproj             src/SolarPipe.Prediction/
COPY src/SolarPipe.Host/SolarPipe.Host.csproj                         src/SolarPipe.Host/

RUN dotnet restore SolarPipe.sln

# Build + publish (Release, linux-x64, self-contained false — use runtime image)
COPY src/ src/
COPY python/solarpipe.proto python/solarpipe.proto

RUN dotnet publish src/SolarPipe.Host/SolarPipe.Host.csproj \
    -c Release \
    -o /app/publish \
    --no-restore

# ─── runtime stage ────────────────────────────────────────────────────────────

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime

WORKDIR /app

# Logs directory (structured JSON to logs/dotnet_latest.json — ADR-010)
RUN mkdir -p logs models/registry

COPY --from=build /app/publish .

# Default: print help. Override CMD to run specific commands.
ENTRYPOINT ["dotnet", "SolarPipe.Host.dll"]
CMD ["--help"]
