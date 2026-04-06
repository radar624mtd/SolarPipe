namespace SolarPipe.Config;

// AST node types for the compose expression grammar:
//   expr  -> chain
//   chain -> residual ('->' residual)*
//   resid -> ensemble ('^' ensemble)*
//   ensem -> atom    ('+' atom)*
//   atom  -> IDENT | '(' expr ')' | gate
//   gate  -> IDENT '?' '(' expr ',' expr ')'
//
// Operators (lowest to highest precedence): -> + ^ ? identifier

public abstract class ComposeNode
{
    public abstract IReadOnlyList<string> ReferencedStages { get; }
}

public sealed class IdentifierNode(string name) : ComposeNode
{
    public string Name { get; } = name;
    public override IReadOnlyList<string> ReferencedStages => [Name];
    public override string ToString() => Name;
}

public sealed class ChainNode(ComposeNode left, ComposeNode right) : ComposeNode
{
    public ComposeNode Left { get; } = left;
    public ComposeNode Right { get; } = right;
    public override IReadOnlyList<string> ReferencedStages =>
        [.. Left.ReferencedStages, .. Right.ReferencedStages];
    public override string ToString() => $"({Left} -> {Right})";
}

public sealed class EnsembleNode(ComposeNode left, ComposeNode right) : ComposeNode
{
    public ComposeNode Left { get; } = left;
    public ComposeNode Right { get; } = right;
    public override IReadOnlyList<string> ReferencedStages =>
        [.. Left.ReferencedStages, .. Right.ReferencedStages];
    public override string ToString() => $"({Left} + {Right})";
}

public sealed class ResidualNode(ComposeNode left, ComposeNode right) : ComposeNode
{
    public ComposeNode Left { get; } = left;
    public ComposeNode Right { get; } = right;
    public override IReadOnlyList<string> ReferencedStages =>
        [.. Left.ReferencedStages, .. Right.ReferencedStages];
    public override string ToString() => $"({Left} ^ {Right})";
}

public sealed class GateNode(ComposeNode classifier, ComposeNode ifTrue, ComposeNode ifFalse) : ComposeNode
{
    public ComposeNode Classifier { get; } = classifier;
    public ComposeNode IfTrue { get; } = ifTrue;
    public ComposeNode IfFalse { get; } = ifFalse;
    public override IReadOnlyList<string> ReferencedStages =>
        [.. Classifier.ReferencedStages, .. IfTrue.ReferencedStages, .. IfFalse.ReferencedStages];
    public override string ToString() => $"({Classifier} ? ({IfTrue}, {IfFalse}))";
}

// Parses compose expressions from pipeline YAML.
// Grammar:
//   chain  -> residual ('->' residual)*          (left-associative, lowest precedence)
//   residual -> ensemble ('^' ensemble)*         (left-associative)
//   ensemble -> gate ('+' gate)*                 (left-associative)
//   gate   -> atom ('?' '(' chain ',' chain ')')?
//   atom   -> IDENT | '(' chain ')'
public sealed class ComposeExpressionParser
{
    private List<Token> _tokens = [];
    private int _pos;

    public ComposeNode Parse(string expression)
    {
        if (string.IsNullOrWhiteSpace(expression))
            throw new ArgumentException(
                "Compose expression is empty. Stage: compose field. Expected: identifier or operator expression.",
                nameof(expression));

        _tokens = Tokenize(expression);
        _pos = 0;

        var node = ParseChain();

        if (_pos < _tokens.Count)
            throw new FormatException(
                $"Unexpected token '{_tokens[_pos].Value}' at position {_tokens[_pos].Position} " +
                $"in compose expression '{expression}'. Expected end of expression.");

        return node;
    }

    // chain -> residual ('->' residual)*
    private ComposeNode ParseChain()
    {
        var left = ParseResidual();

        while (_pos < _tokens.Count && _tokens[_pos].Kind == TokenKind.Arrow)
        {
            _pos++;
            var right = ParseResidual();
            left = new ChainNode(left, right);
        }

        return left;
    }

    // residual -> ensemble ('^' ensemble)*
    private ComposeNode ParseResidual()
    {
        var left = ParseEnsemble();

        while (_pos < _tokens.Count && _tokens[_pos].Kind == TokenKind.Caret)
        {
            _pos++;
            var right = ParseEnsemble();
            left = new ResidualNode(left, right);
        }

        return left;
    }

    // ensemble -> gate ('+' gate)*
    private ComposeNode ParseEnsemble()
    {
        var left = ParseGate();

        while (_pos < _tokens.Count && _tokens[_pos].Kind == TokenKind.Plus)
        {
            _pos++;
            var right = ParseGate();
            left = new EnsembleNode(left, right);
        }

        return left;
    }

    // gate -> atom ('?' '(' chain ',' chain ')')?
    private ComposeNode ParseGate()
    {
        var atom = ParseAtom();

        if (_pos >= _tokens.Count || _tokens[_pos].Kind != TokenKind.Question)
            return atom;

        _pos++; // consume '?'

        Expect(TokenKind.LParen, "'(' after '?' in gate expression");
        var ifTrue = ParseChain();
        Expect(TokenKind.Comma, "',' between gate branches");
        var ifFalse = ParseChain();
        Expect(TokenKind.RParen, "')' to close gate expression");

        return new GateNode(atom, ifTrue, ifFalse);
    }

    // atom -> IDENT | '(' chain ')'
    private ComposeNode ParseAtom()
    {
        if (_pos >= _tokens.Count)
            throw new FormatException(
                "Unexpected end of compose expression. Expected identifier or '('.");

        var tok = _tokens[_pos];

        if (tok.Kind == TokenKind.Identifier)
        {
            _pos++;
            return new IdentifierNode(tok.Value);
        }

        if (tok.Kind == TokenKind.LParen)
        {
            _pos++;
            var inner = ParseChain();
            Expect(TokenKind.RParen, "')' to close grouped expression");
            return inner;
        }

        throw new FormatException(
            $"Unexpected token '{tok.Value}' at position {tok.Position} in compose expression. " +
            $"Expected stage name (identifier) or '('.");
    }

    private void Expect(TokenKind kind, string description)
    {
        if (_pos >= _tokens.Count)
            throw new FormatException(
                $"Unexpected end of compose expression. Expected {description}.");

        if (_tokens[_pos].Kind != kind)
            throw new FormatException(
                $"Expected {description} at position {_tokens[_pos].Position}, " +
                $"got '{_tokens[_pos].Value}'.");

        _pos++;
    }

    private static List<Token> Tokenize(string input)
    {
        var tokens = new List<Token>();
        var i = 0;

        while (i < input.Length)
        {
            if (char.IsWhiteSpace(input[i])) { i++; continue; }

            // Arrow operator '->'
            if (i + 1 < input.Length && input[i] == '-' && input[i + 1] == '>')
            {
                tokens.Add(new Token(TokenKind.Arrow, "->", i));
                i += 2;
                continue;
            }

            switch (input[i])
            {
                case '+': tokens.Add(new Token(TokenKind.Plus, "+", i++)); continue;
                case '^': tokens.Add(new Token(TokenKind.Caret, "^", i++)); continue;
                case '?': tokens.Add(new Token(TokenKind.Question, "?", i++)); continue;
                case '(': tokens.Add(new Token(TokenKind.LParen, "(", i++)); continue;
                case ')': tokens.Add(new Token(TokenKind.RParen, ")", i++)); continue;
                case ',': tokens.Add(new Token(TokenKind.Comma, ",", i++)); continue;
            }

            // Identifier: letters, digits, underscores (stage names follow snake_case/alphanumeric)
            if (char.IsLetter(input[i]) || input[i] == '_')
            {
                var start = i;
                while (i < input.Length && (char.IsLetterOrDigit(input[i]) || input[i] == '_'))
                    i++;
                tokens.Add(new Token(TokenKind.Identifier, input[start..i], start));
                continue;
            }

            throw new FormatException(
                $"Unexpected character '{input[i]}' at position {i} in compose expression '{input}'.");
        }

        return tokens;
    }

    private enum TokenKind { Identifier, Arrow, Plus, Caret, Question, LParen, RParen, Comma }

    private readonly record struct Token(TokenKind Kind, string Value, int Position);
}
