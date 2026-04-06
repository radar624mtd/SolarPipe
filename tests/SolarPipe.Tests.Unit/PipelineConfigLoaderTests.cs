using FluentAssertions;
using SolarPipe.Config;
using YamlDotNet.Core;

namespace SolarPipe.Tests.Unit;

[Trait("Category", "Unit")]
public class PipelineConfigLoaderTests
{
    private readonly PipelineConfigLoader _loader = new();

    private const string ValidYaml = """
        name: test_pipeline
        data_sources:
          src1:
            provider: csv
            connection_string: /data/solar.csv
        stages:
          rf_stage:
            framework: mlnet
            model_type: fast_forest
            data_source: src1
            features:
              - speed
              - density
            target: dst
            hyperparameters:
              number_of_trees: 100
        """;

    // Test 1: Load valid YAML configuration
    [Fact]
    public void LoadFromString_ValidYaml_ReturnsPipelineConfig()
    {
        var config = _loader.LoadFromString(ValidYaml);

        config.Name.Should().Be("test_pipeline");
        config.DataSources.Should().ContainKey("src1");
        config.Stages.Should().ContainKey("rf_stage");
        config.Stages["rf_stage"].Features.Should().BeEquivalentTo(["speed", "density"]);
        config.Stages["rf_stage"].Target.Should().Be("dst");
    }

    // Test 2: Validate missing data source error
    [Fact]
    public void LoadFromString_StageReferencesUnknownDataSource_Throws()
    {
        var yaml = """
            name: bad
            data_sources:
              real_source:
                provider: csv
                connection_string: /data/test.csv
            stages:
              s1:
                framework: mlnet
                model_type: fast_forest
                data_source: nonexistent_source
                features: [speed]
                target: dst
            """;

        var act = () => _loader.LoadFromString(yaml);
        act.Should().Throw<ArgumentException>()
            .WithMessage("*nonexistent_source*");
    }

    // Test 3: Validate missing features error
    [Fact]
    public void LoadFromString_StageWithNoFeatures_Throws()
    {
        var yaml = """
            name: bad
            data_sources:
              src1:
                provider: csv
                connection_string: /data/test.csv
            stages:
              s1:
                framework: mlnet
                model_type: fast_forest
                data_source: src1
                features: []
                target: dst
            """;

        var act = () => _loader.LoadFromString(yaml);
        act.Should().Throw<ArgumentException>()
            .WithMessage("*no features*");
    }

    // Test 4: Deserialize complex nested structures including hyperparameters
    [Fact]
    public void LoadFromString_HyperparametersDeserialized_Correctly()
    {
        var config = _loader.LoadFromString(ValidYaml);

        config.Stages["rf_stage"].Hyperparameters.Should().NotBeNull();
        config.Stages["rf_stage"].Hyperparameters!.Should().ContainKey("number_of_trees");
    }

    // Test 5: Handle missing optional fields (no evaluation, no output, no compose)
    [Fact]
    public void LoadFromString_OptionalFieldsAbsent_LoadsWithDefaults()
    {
        var yaml = """
            name: minimal
            data_sources:
              s:
                provider: sqlite
                connection_string: /db/solar.db
            stages:
              stage1:
                framework: mlnet
                model_type: fast_forest
                data_source: s
                features: [bz]
                target: dst
            """;

        var config = _loader.LoadFromString(yaml);

        config.Compose.Should().BeNull();
        config.Evaluation.Should().BeNull();
        config.Output.Should().BeNull();
    }

    // Test 6: YAML 1.2 boolean enforcement — NO/yes/on/off must throw (RULE-020)
    [Theory]
    [InlineData("NO")]
    [InlineData("yes")]
    [InlineData("on")]
    [InlineData("off")]
    [InlineData("YES")]
    [InlineData("On")]
    public void Yaml12BooleanConverter_Yaml11Literals_ThrowYamlException(string value)
    {
        var converter = new Yaml12BooleanConverter();
        // Parser starts before StreamStart; need to advance past StreamStart + DocumentStart
        var parser = new YamlDotNet.Core.Parser(new StringReader(value));
        parser.Consume<YamlDotNet.Core.Events.StreamStart>();
        parser.Consume<YamlDotNet.Core.Events.DocumentStart>();

        var act = () => converter.ReadYaml(parser, typeof(bool));
        act.Should().Throw<YamlDotNet.Core.YamlException>()
            .WithMessage($"*{value}*");
    }

    // Test 7: YAML 1.2 booleans — true/false/True/False/TRUE/FALSE are accepted
    [Theory]
    [InlineData("true", true)]
    [InlineData("false", false)]
    [InlineData("True", true)]
    [InlineData("False", false)]
    [InlineData("TRUE", true)]
    [InlineData("FALSE", false)]
    public void Yaml12BooleanConverter_ValidLiterals_Parsed(string literal, bool expected)
    {
        var converter = new Yaml12BooleanConverter();
        var parser = new YamlDotNet.Core.Parser(new StringReader(literal));
        parser.Consume<YamlDotNet.Core.Events.StreamStart>();
        parser.Consume<YamlDotNet.Core.Events.DocumentStart>();

        var result = converter.ReadYaml(parser, typeof(bool));
        result.Should().Be(expected);
    }

    // Test 8: RULE-021 null rejection — required object property null throws
    // (DataSourceYaml.Options is nullable, but DataSources dict value itself is required)
    // We test this by providing an explicit null value for a required string field.
    [Fact]
    public void LoadFromString_ExplicitNullOnRequiredProperty_ThrowsInvalidOperation()
    {
        // Explicitly set provider to YAML null scalar — YamlDotNet will assign null
        // to the non-nullable Provider string (GitHub #763 behaviour)
        var yaml = """
            name: nulltest
            data_sources:
              src:
                provider: ~
                connection_string: /data/test.csv
            stages: {}
            """;

        var act = () => _loader.LoadFromString(yaml);
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*Provider*null*");
    }

    // Test 9: Quoted string passthrough — "NO" and "yes" survive as strings in object fields
    [Fact]
    public void LoadFromString_QuotedNorwayValues_SurviveAsStrings()
    {
        var yaml = """
            name: quoted_test
            data_sources:
              src:
                provider: csv
                connection_string: /data/test.csv
                options:
                  coordinate_frame: "NO"
                  direction: "yes"
            stages:
              s1:
                framework: mlnet
                model_type: fast_forest
                data_source: src
                features: [bz]
                target: dst
            """;

        var config = _loader.LoadFromString(yaml);

        config.DataSources["src"].Options.Should().NotBeNull();
        config.DataSources["src"].Options!["coordinate_frame"].Should().Be("NO");
        config.DataSources["src"].Options!["direction"].Should().Be("yes");
    }

    // Test 10: LoadAsync reads from file path correctly
    [Fact]
    public async Task LoadAsync_ValidFile_ReturnsPipelineConfig()
    {
        var fixturePath = Path.Combine(
            Path.GetDirectoryName(typeof(PipelineConfigLoaderTests).Assembly.Location)!,
            "..", "..", "..", "..", "..", "tests", "fixtures", "sample_config.yaml");
        fixturePath = Path.GetFullPath(fixturePath);

        if (!File.Exists(fixturePath))
        {
            // Write a temp fixture if the relative path resolution fails
            var tmp = Path.GetTempFileName() + ".yaml";
            await File.WriteAllTextAsync(tmp, ValidYaml);
            var config2 = await _loader.LoadAsync(tmp, CancellationToken.None);
            config2.Name.Should().Be("test_pipeline");
            File.Delete(tmp);
            return;
        }

        var config = await _loader.LoadAsync(fixturePath, CancellationToken.None);
        config.Name.Should().Be("test_pipeline");
        config.DataSources.Should().NotBeEmpty();
        config.Stages.Should().NotBeEmpty();
    }
}
