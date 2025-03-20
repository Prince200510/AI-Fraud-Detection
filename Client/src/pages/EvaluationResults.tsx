import React from "react";

type ClassificationReport = {
  precision: number;
  recall: number;
  "f1-score": number;
  support: number;
};

type EvaluationData = {
  rf_report: {
    [key: string]: ClassificationReport;
    accuracy: number;
  };
  rf_conf_matrix: number[][];
  test_accuracy?: number;
  fraud_detected_test?: number;
  total_fraud_present_test?: number;
};

type EvaluationResultsProps = {
  evaluation?: EvaluationData;
};

const EvaluationResults: React.FC<EvaluationResultsProps> = ({ evaluation }) => {
  if (!evaluation) return <p>No evaluation data available.</p>;

  const { rf_report, rf_conf_matrix, test_accuracy, fraud_detected_test, total_fraud_present_test } = evaluation;

  return (
    <div className="evaluation-container">
      <h2>📊 Model Evaluation</h2>

      {/* Classification Report */}
      <div>
        <h3>🔹 Classification Report</h3>
        <table>
          <thead>
            <tr>
              <th>Class</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-Score</th>
              <th>Support</th>
            </tr>
          </thead>
          <tbody>
            {["0", "1"].map((label) => (
              <tr key={label}>
                <td>{label}</td>
                <td>{rf_report[label].precision.toFixed(4)}</td>
                <td>{rf_report[label].recall.toFixed(4)}</td>
                <td>{rf_report[label]["f1-score"].toFixed(4)}</td>
                <td>{rf_report[label].support}</td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <td>Accuracy</td>
              <td colSpan={4}>{(rf_report.accuracy * 100).toFixed(2)}%</td>
            </tr>
          </tfoot>
        </table>
      </div>

      {/* Confusion Matrix */}
      <div>
        <h3>📊 Confusion Matrix</h3>
        <table className="confusion-matrix">
          <thead>
            <tr>
              <th></th>
              <th colSpan={2}>Predicted</th>
            </tr>
            <tr>
              <th>Actual</th>
              <th>Class 0</th>
              <th>Class 1</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>Class 0</th>
              <td>{rf_conf_matrix[0][0]}</td>
              <td>{rf_conf_matrix[0][1]}</td>
            </tr>
            <tr>
              <th>Class 1</th>
              <td>{rf_conf_matrix[1][0]}</td>
              <td>{rf_conf_matrix[1][1]}</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Additional Evaluation Metrics */}
      {test_accuracy !== undefined && (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">📈 Additional Metrics</h3>
          <p className="text-gray-600 dark:text-gray-400">
            <strong>Test Accuracy:</strong> {test_accuracy.toFixed(2)}
          </p>
          <p className="text-gray-600 dark:text-gray-400">
            <strong>Fraud Detected:</strong> {fraud_detected_test} / {total_fraud_present_test}
          </p>
        </div>
      )}
    </div>
  );
};

export default EvaluationResults;
