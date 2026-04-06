using YamlDotNet.Core;
using YamlDotNet.Core.Events;
using YamlDotNet.Serialization;

namespace SolarPipe.Config;

// RULE-020: Enforces YAML 1.2 boolean semantics.
// YamlDotNet 15.x implements YAML 1.1, which treats yes/no/on/off as booleans
// (the "Norway problem"). This converter restricts booleans to YAML 1.2 literals only.
public class Yaml12BooleanConverter : IYamlTypeConverter
{
    private static readonly HashSet<string> TrueValues =
        new(StringComparer.Ordinal) { "true", "True", "TRUE" };
    private static readonly HashSet<string> FalseValues =
        new(StringComparer.Ordinal) { "false", "False", "FALSE" };

    public bool Accepts(Type type) => type == typeof(bool) || type == typeof(bool?);

    public object? ReadYaml(IParser parser, Type type)
    {
        var scalar = parser.Consume<Scalar>();
        if (TrueValues.Contains(scalar.Value)) return true;
        if (FalseValues.Contains(scalar.Value)) return false;
        if (type == typeof(bool?)) return null;
        throw new YamlException(scalar.Start, scalar.End,
            $"Invalid boolean '{scalar.Value}'. " +
            $"YAML 1.2 allows only true/false/True/False/TRUE/FALSE. " +
            $"Wrap in quotes if this is a string value.");
    }

    public void WriteYaml(IEmitter emitter, object? value, Type type)
    {
        emitter.Emit(new Scalar(value is true ? "true" : "false"));
    }
}
