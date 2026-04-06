namespace SolarPipe.Core.Models;

public enum FrameworkType { MlNet, Onnx, Physics, PythonGrpc }

public enum TaskType { Regression, Classification, AnomalyDetection }

public enum ColumnRole { Feature, Target, Timestamp, Identifier, Auxiliary }
