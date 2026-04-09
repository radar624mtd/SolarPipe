using SolarPipe.Config;
using SolarPipe.Config.Models;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data;
using SolarPipe.Data.DataFrame;
using SolarPipe.Prediction;
using SolarPipe.Training.Validation;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace SolarPipe.Host.Commands;

public sealed class ValidateEventsCommand : ICommand
{
    private readonly IDeserializer _yaml = new DeserializerBuilder()
        .WithNamingConvention(UnderscoredNamingConvention.Instance)
        .Build();

    private readonly PipelineConfigLoader _pipelineLoader;
    private readonly IModelRegistry _modelRegistry;
    private readonly DataSourceRegistry _dataRegistry;

    public ValidateEventsCommand(
        PipelineConfigLoader pipelineLoader,
        IModelRegistry modelRegistry,
        DataSourceRegistry dataRegistry)
    {
        _pipelineLoader = pipelineLoader;
        _modelRegistry = modelRegistry;
        _dataRegistry = dataRegistry;
    }

    public async Task<int> ExecuteAsync(string[] args, CancellationToken ct)
    {
        string configPath;
        try { configPath = ArgParser.Require(args, "--config"); }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine($"VALIDATE_EVENTS_ERROR type=MissingArguments message=\"{ex.Message}\"");
            return ExitCodes.MissingArguments;
        }

        var outputPath = ArgParser.Get(args, "--output");

        try
        {
            if (!File.Exists(configPath))
                throw new FileNotFoundException($"Validate config not found: {configPath}");

            var yaml = await File.ReadAllTextAsync(configPath, ct);
            var config = _yaml.Deserialize<ValidateConfig>(yaml)
                ?? throw new InvalidOperationException("validate_2026.yaml deserialized to null.");

            var observed  = await EventValidator.LoadObservedTransitsAsync(config.ObservedTransits, ct);
            var scoreboard = await EventValidator.LoadScoreboardAsync(config.Scoreboard, ct);

            var predictions = await RunPredictionsAsync(config, observed, ct);

            var report = EventValidator.BuildReport(observed, scoreboard, predictions);

            Console.WriteLine(report.AsciiTable);

            var outFile = outputPath ?? Path.Combine("output",
                $"validation_{DateTime.UtcNow:yyyyMMdd_HHmmss}.json");
            Directory.CreateDirectory(Path.GetDirectoryName(outFile)!);
            await File.WriteAllTextAsync(outFile, EventValidator.SerializeReport(report), ct);

            Console.WriteLine($"VALIDATE_EVENTS_OK events={observed.Count} predictions={predictions.Count} output={outFile}");
            return ExitCodes.Success;
        }
        catch (FileNotFoundException ex)
        {
            Console.Error.WriteLine(
                $"VALIDATE_EVENTS_ERROR type=FileNotFound message=\"{ex.Message}\"");
            return ExitCodes.ValidationFailed;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(
                $"VALIDATE_EVENTS_ERROR type={ex.GetType().Name} message=\"{ex.Message}\"");
            return ExitCodes.ValidationFailed;
        }
    }

    private async Task<List<EventPrediction>> RunPredictionsAsync(
        ValidateConfig config,
        IReadOnlyList<ObservedTransit> observed,
        CancellationToken ct)
    {
        var predictions = new List<EventPrediction>();

        if (string.IsNullOrWhiteSpace(config.PipelineConfig))
            return predictions;

        try
        {
            var pipelineConfig = await _pipelineLoader.LoadAsync(config.PipelineConfig, ct);

            // Register data sources so we can query the catalog for full feature vectors
            foreach (var (key, _) in pipelineConfig.DataSources)
                _dataRegistry.RegisterSource(pipelineConfig.ToDataSourceConfig(key));

            // Load both stages: physics baseline + ML correction
            ITrainedModel? baselineModel = null;
            ITrainedModel? correctionModel = null;
            string? correctionStageName = null;
            StageConfig? correctionStageConfig = null;

            foreach (var (stageName, stageYaml) in pipelineConfig.Stages)
            {
                var artifacts = await _modelRegistry.ListAsync(stageName, ct);
                if (artifacts.Count == 0)
                {
                    Console.Error.WriteLine(
                        $"VALIDATE_EVENTS_WARN stage={stageName} message=\"No trained model. Run train first.\"");
                    continue;
                }

                var latest = artifacts.OrderByDescending(a => a.TrainedAt).First();
                var model = await _modelRegistry.LoadAsync(latest.ModelId, latest.Version, ct);

                if (string.Equals(stageYaml.Framework, "Physics", StringComparison.OrdinalIgnoreCase))
                {
                    baselineModel = model;
                    Console.WriteLine(
                        $"VALIDATE_EVENTS_LOAD stage={stageName} model={latest.ModelId}");
                }
                else
                {
                    correctionModel = model;
                    correctionStageName = stageName;
                    correctionStageConfig = pipelineConfig.ToStageConfig(stageName);
                    Console.WriteLine(
                        $"VALIDATE_EVENTS_LOAD stage={stageName} model={latest.ModelId}");
                }
            }

            if (baselineModel is null)
                return predictions;

            // Build input DataFrame from the catalog DB — same source as training
            InMemoryDataFrame inputFrame;
            if (correctionStageConfig is not null)
            {
                var corrDsName = pipelineConfig.Stages
                    .First(s => !string.Equals(s.Value.Framework, "Physics", StringComparison.OrdinalIgnoreCase))
                    .Value.DataSource;
                inputFrame = await BuildValidationFrameFromDbAsync(
                    observed, correctionStageConfig, corrDsName, ct);
            }
            else
            {
                var speeds = observed.Select(o => (float)o.CmeSpeedKms).ToArray();
                var schema = new DataSchema([new ColumnInfo("cme_speed_kms", ColumnType.Float, false)]);
                inputFrame = new InMemoryDataFrame(schema, [speeds]);
            }

            PredictionResult result;
            if (correctionModel is not null)
            {
                // Compose: baseline ^ correction via ResidualModel
                var composed = new ResidualModel(baselineModel, correctionModel, "validation_composed");
                result = await composed.PredictAsync(inputFrame, ct);
                Console.WriteLine(
                    $"VALIDATE_EVENTS_PREDICT mode=composed (drag_baseline^{correctionStageName}) events={observed.Count}");
            }
            else
            {
                // Fallback: physics-only
                result = await baselineModel.PredictAsync(inputFrame, ct);
                Console.WriteLine(
                    $"VALIDATE_EVENTS_PREDICT mode=physics_only events={observed.Count}");
            }

            inputFrame.Dispose();

            // Calibrate conformal prediction intervals from training residuals
            float[]? lowerBounds = null;
            float[]? upperBounds = null;
            try
            {
                var enbpi = await CalibrateConformalAsync(pipelineConfig, baselineModel,
                    correctionModel, correctionStageConfig, ct);
                if (enbpi is not null)
                {
                    (lowerBounds, upperBounds) = enbpi.GetIntervals(result.Values);
                    Console.WriteLine(
                        $"VALIDATE_EVENTS_CONFORMAL window={enbpi.WindowCount} " +
                        $"width_90={enbpi.GetIntervalWidth():F1}h");
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(
                    $"VALIDATE_EVENTS_WARN conformal_error type={ex.GetType().Name} message=\"{ex.Message}\"");
            }

            for (int i = 0; i < observed.Count; i++)
            {
                if (i < result.Values.Length && !float.IsNaN(result.Values[i]))
                {
                    double lo = lowerBounds is not null ? lowerBounds[i] : double.NaN;
                    double hi = upperBounds is not null ? upperBounds[i] : double.NaN;
                    predictions.Add(new EventPrediction(observed[i].CmeId, result.Values[i], lo, hi));
                }
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(
                $"VALIDATE_EVENTS_WARN predict_error type={ex.GetType().Name} message=\"{ex.Message}\"");
        }

        return predictions;
    }

    private async Task<InMemoryDataFrame> BuildValidationFrameFromDbAsync(
        IReadOnlyList<ObservedTransit> observed,
        StageConfig correctionConfig,
        string dsName,
        CancellationToken ct)
    {
        int nFeatures = correctionConfig.Features.Count;
        var allCols = new float[nFeatures][];
        for (int f = 0; f < nFeatures; f++)
            allCols[f] = new float[observed.Count];

        string sql = BuildValidationEventSql(correctionConfig);
        int dbHits = 0;

        for (int i = 0; i < observed.Count; i++)
        {
            var parameters = new Dictionary<string, object> { ["eventId"] = observed[i].CmeId };
            var query = new DataQuery(Sql: sql, Parameters: parameters);

            try
            {
                using var frame = await _dataRegistry.LoadAsync(dsName, query, ct);
                if (frame.RowCount > 0)
                {
                    for (int f = 0; f < nFeatures && f < frame.Schema.Columns.Count; f++)
                        allCols[f][i] = frame.GetColumn(f)[0];
                    dbHits++;
                }
                else
                {
                    for (int f = 0; f < nFeatures; f++)
                        allCols[f][i] = float.NaN;
                    Console.Error.WriteLine(
                        $"VALIDATE_EVENTS_WARN event={observed[i].CmeId} " +
                        "message=\"Not found in cme_catalog.db; features will be NaN\"");
                }
            }
            catch (Exception ex)
            {
                for (int f = 0; f < nFeatures; f++)
                    allCols[f][i] = float.NaN;
                Console.Error.WriteLine(
                    $"VALIDATE_EVENTS_WARN event={observed[i].CmeId} " +
                    $"message=\"DB query failed: {ex.Message}\"");
            }
        }

        Console.WriteLine(
            $"VALIDATE_EVENTS_FEATURES source=cme_catalog.db db_hits={dbHits}/{observed.Count}");

        var schema = new DataSchema(
            correctionConfig.Features
                .Select(f => new ColumnInfo(f, ColumnType.Float, false)).ToList());
        return new InMemoryDataFrame(schema, allCols);
    }

    private static string BuildValidationEventSql(StageConfig stage)
    {
        // Query raw tables directly (not training_features view) to include events
        // where transit_time_hours is NULL — we have ground truth from the CSV
        var selectParts = stage.Features.Select(f => f.ToLowerInvariant() switch
        {
            "cme_speed_kms"        => "e.\"cme_speed\" AS \"cme_speed_kms\"",
            "bz_gsm_proxy_nt"      => "f.\"observed_bz_min\" AS \"bz_gsm_proxy_nt\"",
            "sw_density_n_cc"      => "e.\"sw_density_ambient\" AS \"sw_density_n_cc\"",
            "sw_speed_ambient_kms" => "e.\"sw_speed_ambient\" AS \"sw_speed_ambient_kms\"",
            "sw_bt_nt"             => "e.\"sw_bt_ambient\" AS \"sw_bt_nt\"",
            "delta_v_kms"          => "(e.\"cme_speed\" - e.\"sw_speed_ambient\") AS \"delta_v_kms\"",
            "speed_ratio"          => "(e.\"cme_speed\" / NULLIF(e.\"sw_speed_ambient\", 0)) AS \"speed_ratio\"",
            "speed_x_bz"           => "(e.\"cme_speed\" * f.\"observed_bz_min\") AS \"speed_x_bz\"",
            "speed_x_density"      => "(e.\"cme_speed\" * e.\"sw_density_ambient\") AS \"speed_x_density\"",
            _ => $"\"{f.Replace("\"", "")}\""
        });

        return $"SELECT {string.Join(", ", selectParts)} " +
               "FROM \"cme_events\" e " +
               "JOIN \"flux_rope_fits\" f ON e.\"event_id\" = f.\"event_id\" " +
               "WHERE e.\"event_id\" = @eventId AND e.\"cme_speed\" IS NOT NULL";
    }
    private async Task<EnbPiPredictor?> CalibrateConformalAsync(
        PipelineConfig pipelineConfig,
        ITrainedModel baselineModel,
        ITrainedModel? correctionModel,
        StageConfig? correctionStageConfig,
        CancellationToken ct)
    {
        // Load training data from the catalog to compute calibration residuals
        var correctionStageYaml = pipelineConfig.Stages
            .FirstOrDefault(s => !string.Equals(s.Value.Framework, "Physics", StringComparison.OrdinalIgnoreCase));
        if (correctionStageYaml.Key is null)
            return null;

        var dsName = correctionStageYaml.Value.DataSource;
        var dsOptions = pipelineConfig.DataSources[dsName].Options;
        string table = dsOptions is not null
            && dsOptions.TryGetValue("table", out var t)
            && !string.IsNullOrWhiteSpace(t)
            ? t : dsName;

        // Load training features + target for calibration
        var stageConfig = correctionStageConfig ?? pipelineConfig.ToStageConfig(correctionStageYaml.Key);
        string sql = BuildCalibrationSql(stageConfig, table);

        var data = await _dataRegistry.LoadAsync(dsName, new DataQuery(Sql: sql), ct);
        if (data.RowCount < 10)
        {
            data.Dispose();
            return null;
        }

        // Run composed model on training data to get predictions
        var targetCol = data.GetColumn(stageConfig.Target);

        PredictionResult calResult;
        if (correctionModel is not null)
        {
            var composed = new ResidualModel(baselineModel, correctionModel, "calibration");
            calResult = await composed.PredictAsync(data, ct);
        }
        else
        {
            calResult = await baselineModel.PredictAsync(data, ct);
        }
        data.Dispose();

        // Calibrate EnbPI from training residuals (temporal order preserved)
        var enbpi = new EnbPiPredictor(windowSize: Math.Min(100, calResult.Values.Length));
        enbpi.Calibrate(targetCol, calResult.Values);

        return enbpi;
    }

    private static string BuildCalibrationSql(StageConfig stage, string table)
    {
        static string Q(string c) => $"\"{c.Replace("\"", "")}\"";

        var features = stage.Features.ToList();
        var allCols = features.Append(stage.Target)
            .Distinct(StringComparer.OrdinalIgnoreCase);

        var selectParts = allCols.Select(f =>
        {
            if (string.Equals(f, "delta_v_kms", StringComparison.OrdinalIgnoreCase))
                return $"({Q("cme_speed_kms")} - {Q("sw_speed_ambient_kms")}) AS {Q("delta_v_kms")}";
            if (string.Equals(f, "speed_ratio", StringComparison.OrdinalIgnoreCase))
                return $"({Q("cme_speed_kms")} / NULLIF({Q("sw_speed_ambient_kms")}, 0)) AS {Q("speed_ratio")}";
            if (string.Equals(f, "speed_x_bz", StringComparison.OrdinalIgnoreCase))
                return $"({Q("cme_speed_kms")} * {Q("bz_gsm_proxy_nt")}) AS {Q("speed_x_bz")}";
            if (string.Equals(f, "speed_x_density", StringComparison.OrdinalIgnoreCase))
                return $"({Q("cme_speed_kms")} * {Q("sw_density_n_cc")}) AS {Q("speed_x_density")}";
            return Q(f);
        });

        return $"SELECT {string.Join(", ", selectParts)} FROM {Q(table)}";
    }
}

internal sealed class ValidateConfig
{
    public string PipelineConfig { get; set; } = "";
    public string ObservedTransits { get; set; } = "";
    public string Scoreboard { get; set; } = "";
}
