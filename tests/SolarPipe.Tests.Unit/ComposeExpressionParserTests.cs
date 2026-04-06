using FluentAssertions;
using SolarPipe.Config;

namespace SolarPipe.Tests.Unit;

[Trait("Category", "Unit")]
public class ComposeExpressionParserTests
{
    private readonly ComposeExpressionParser _parser = new();

    // Test 1: Single identifier parses to IdentifierNode
    [Fact]
    public void Parse_SingleIdentifier_ReturnsIdentifierNode()
    {
        var node = _parser.Parse("rf_baseline");

        node.Should().BeOfType<IdentifierNode>();
        ((IdentifierNode)node).Name.Should().Be("rf_baseline");
    }

    // Test 2: Chain operator '->' produces ChainNode (left-associative)
    [Fact]
    public void Parse_ChainOperator_ProducesChainNode()
    {
        var node = _parser.Parse("physics_model -> rf_correction");

        node.Should().BeOfType<ChainNode>();
        var chain = (ChainNode)node;
        ((IdentifierNode)chain.Left).Name.Should().Be("physics_model");
        ((IdentifierNode)chain.Right).Name.Should().Be("rf_correction");
    }

    // Test 3: Ensemble operator '+' produces EnsembleNode
    [Fact]
    public void Parse_EnsembleOperator_ProducesEnsembleNode()
    {
        var node = _parser.Parse("model_a + model_b");

        node.Should().BeOfType<EnsembleNode>();
        var ensemble = (EnsembleNode)node;
        ((IdentifierNode)ensemble.Left).Name.Should().Be("model_a");
        ((IdentifierNode)ensemble.Right).Name.Should().Be("model_b");
    }

    // Test 4: Residual operator '^' produces ResidualNode (physics baseline ^ ML correction)
    [Fact]
    public void Parse_ResidualOperator_ProducesResidualNode()
    {
        var node = _parser.Parse("drag_physics ^ regression_forest");

        node.Should().BeOfType<ResidualNode>();
        var residual = (ResidualNode)node;
        ((IdentifierNode)residual.Left).Name.Should().Be("drag_physics");
        ((IdentifierNode)residual.Right).Name.Should().Be("regression_forest");
    }

    // Test 5: Gate operator '?' produces GateNode with classifier + two branches
    [Fact]
    public void Parse_GateOperator_ProducesGateNode()
    {
        var node = _parser.Parse("storm_classifier ? (storm_model, quiet_model)");

        node.Should().BeOfType<GateNode>();
        var gate = (GateNode)node;
        ((IdentifierNode)gate.Classifier).Name.Should().Be("storm_classifier");
        ((IdentifierNode)gate.IfTrue).Name.Should().Be("storm_model");
        ((IdentifierNode)gate.IfFalse).Name.Should().Be("quiet_model");
    }

    // Test 6: Operator precedence — '^' binds tighter than '->'
    // "a -> b ^ c" should parse as "a -> (b ^ c)"
    [Fact]
    public void Parse_ResidualBindsTighterThanChain_CorrectPrecedence()
    {
        var node = _parser.Parse("physics_model -> rf_stage ^ correction");

        node.Should().BeOfType<ChainNode>();
        var chain = (ChainNode)node;
        ((IdentifierNode)chain.Left).Name.Should().Be("physics_model");
        chain.Right.Should().BeOfType<ResidualNode>();
        var residual = (ResidualNode)chain.Right;
        ((IdentifierNode)residual.Left).Name.Should().Be("rf_stage");
        ((IdentifierNode)residual.Right).Name.Should().Be("correction");
    }

    // Test 7: Operator precedence — '+' binds tighter than '^'
    // "a ^ b + c" should parse as "a ^ (b + c)"
    [Fact]
    public void Parse_EnsembleBindsTighterThanResidual_CorrectPrecedence()
    {
        var node = _parser.Parse("baseline ^ model_a + model_b");

        node.Should().BeOfType<ResidualNode>();
        var residual = (ResidualNode)node;
        ((IdentifierNode)residual.Left).Name.Should().Be("baseline");
        residual.Right.Should().BeOfType<EnsembleNode>();
        var ensemble = (EnsembleNode)residual.Right;
        ((IdentifierNode)ensemble.Left).Name.Should().Be("model_a");
        ((IdentifierNode)ensemble.Right).Name.Should().Be("model_b");
    }

    // Test 8: Parentheses override precedence
    // "(a -> b) ^ c" should parse as ResidualNode(ChainNode(a,b), c)
    [Fact]
    public void Parse_ParenthesesOverridePrecedence_CorrectGrouping()
    {
        var node = _parser.Parse("(physics_model -> preprocess) ^ rf_correction");

        node.Should().BeOfType<ResidualNode>();
        var residual = (ResidualNode)node;
        residual.Left.Should().BeOfType<ChainNode>();
        var chain = (ChainNode)residual.Left;
        ((IdentifierNode)chain.Left).Name.Should().Be("physics_model");
        ((IdentifierNode)chain.Right).Name.Should().Be("preprocess");
        ((IdentifierNode)residual.Right).Name.Should().Be("rf_correction");
    }

    // Test 9: ReferencedStages collects all stage names in the expression
    [Fact]
    public void Parse_ComplexExpression_ReferencedStagesContainsAll()
    {
        var node = _parser.Parse("physics -> rf_stage ^ correction");

        node.ReferencedStages.Should().BeEquivalentTo(["physics", "rf_stage", "correction"]);
    }

    // Test 10: Left-associative chain — "a -> b -> c" is "(a -> b) -> c"
    [Fact]
    public void Parse_MultiChain_IsLeftAssociative()
    {
        var node = _parser.Parse("stage_a -> stage_b -> stage_c");

        node.Should().BeOfType<ChainNode>();
        var outer = (ChainNode)node;
        outer.Left.Should().BeOfType<ChainNode>(); // (a -> b) is the left subtree
        var inner = (ChainNode)outer.Left;
        ((IdentifierNode)inner.Left).Name.Should().Be("stage_a");
        ((IdentifierNode)inner.Right).Name.Should().Be("stage_b");
        ((IdentifierNode)outer.Right).Name.Should().Be("stage_c");
    }

    // Test 11: Error — empty expression throws ArgumentException
    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    public void Parse_EmptyExpression_ThrowsArgumentException(string input)
    {
        var act = () => _parser.Parse(input);
        act.Should().Throw<ArgumentException>()
            .WithMessage("*empty*");
    }

    // Test 12: Error cases — malformed expressions throw FormatException
    [Theory]
    [InlineData("->", "starts with arrow, no left operand")]
    [InlineData("a ^", "missing right operand after '^'")]
    [InlineData("a ? (b)", "gate missing comma and second branch")]
    [InlineData("a ? (b, c", "gate missing closing paren")]
    [InlineData("a b", "two identifiers with no operator")]
    [InlineData("a $ b", "unknown character '$'")]
    public void Parse_MalformedExpressions_ThrowFormatException(string input, string reason)
    {
        _ = reason;
        var act = () => _parser.Parse(input);
        act.Should().Throw<FormatException>();
    }
}
