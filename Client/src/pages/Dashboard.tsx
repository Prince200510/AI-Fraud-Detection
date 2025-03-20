import { useSelector } from "react-redux";
import type { RootState } from "../store";
import { useState } from "react";
import axios from "axios";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";

const Dashboard = () => {
  const { isLoading, error } = useSelector(
    (state: RootState) => state.dashboard
  );
  const [classificationReport, setClassificationReport] = useState<any>(null);
const [confusionMatrix, setConfusionMatrix] = useState<number[][] | null>(null);



const [classificationReport1, setClassificationReport1] = useState<any>(null);
const [confusionMatrix1, setConfusionMatrix1] = useState<number[][] | null>(null);
const [accuracy, setAccuracy] = useState<number>(null);
const [recall, setRecall] = useState<number | null>(null);
const [f1Score, setF1Score] = useState<number | null>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [fraudCounts, setFraudCounts] = useState({
    rf: 0,
    lr: 0,
    nn: 0,
    log: 0, // Added for logistic regression fraud count
  });

  const [fraudCounts1, setFraudCounts1] = useState({
    rf: 0,
    lr: 0,
    nn: 0,
    log: 0, // Added for logistic regression fraud count
  });

  const [evaluationMetrics, setEvaluationMetrics] = useState<{
    test_accuracy?: number;
    fraud_detected_test?: number;
    total_fraud_present_test?: number;
  } | null>(null);

  const [evaluationMetrics1, setEvaluationMetrics1] = useState<{
    test_accuracy?: number;
    fraud_detected_test?: number;
    total_fraud_present_test?: number;
  } | null>(null);


  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      alert("Please select a file first");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      // Send file to all models
      const [rfRes, lrRes, nnRes, logRes, evalRes, evalResnn] = await Promise.all([
        axios.post("http://localhost:8000/predict/rf/", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        }),
        axios.post("http://localhost:8000/predict/lr/", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        }),
        axios.post("http://localhost:8000/predict/nn/", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        }),
        axios.post("http://localhost:8000/predict/log/", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        }),
        axios.post("http://localhost:8000/evaluate/", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        }),
        axios.post("http://localhost:8000/evaluate_nn/", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        }),
      ]);

      console.log("📊 Evaluation Response:", evalRes.data);
      console.log("📊 Evaluation Response for:", evalResnn.data);

      setClassificationReport1(evalResnn.data.classification_report);
      setConfusionMatrix1(evalResnn.data.confusion_matrix);
      setAccuracy(evalResnn.data.accuracy);
      setRecall(evalResnn.data.recall);
      setF1Score(evalResnn.data.f1_score);
      setShowResults(true);
      // const { accuracy, recall, f1_score, classification_report, confusion_matrix } = evalResnn.data;

      // if (accuracy && recall && f1_score && classification_report && confusion_matrix) {
      
      // }
      
      console.log("classificationReport1:", classificationReport1);
      console.log("confusionMatrix1:", confusionMatrix1);        
     

      const { rf_report, rf_conf_matrix } = evalRes.data;

      if (rf_report && rf_conf_matrix) {
        setClassificationReport(rf_report);
        setConfusionMatrix(rf_conf_matrix);
      }

      if (rf_report && rf_conf_matrix) {
        console.log("\n🔹 Classification Report:");
        console.table({
          "Class 0": {
            Precision: rf_report["0"].precision.toFixed(4),
            Recall: rf_report["0"].recall.toFixed(4),
            "F1-Score": rf_report["0"]["f1-score"].toFixed(4),
            Support: rf_report["0"].support,
          },
          "Class 1": {
            Precision: rf_report["1"].precision.toFixed(4),
            Recall: rf_report["1"].recall.toFixed(4),
            "F1-Score": rf_report["1"]["f1-score"].toFixed(4),
            Support: rf_report["1"].support,
          },
          Accuracy: {
            Precision: "-",
            Recall: "-",
            "F1-Score": "-",
            Support: `${(rf_report.accuracy * 100).toFixed(2)}%`,
          },
        });
      
        console.log("\n📊 Confusion Matrix:");
        console.table([
          ["Actual \\ Predicted", "Class 0", "Class 1"],
          ["Class 0", rf_conf_matrix[0][0], rf_conf_matrix[0][1]],
          ["Class 1", rf_conf_matrix[1][0], rf_conf_matrix[1][1]],
        ]);
      
        console.log("\n📌 Actual vs. Predicted Values for Class 1:");
        console.log(`✅ True Positives (TP): ${rf_conf_matrix[1][1]}`);
        console.log(`❌ False Negatives (FN): ${rf_conf_matrix[1][0]}`);
      }

      
      setFraudCounts({
        rf: rfRes.data.rf_fraud_count,
        lr: lrRes.data.lr_fraud_count,
        nn: nnRes.data.nn_fraud_count,
        log: logRes.data.fraud_detected_test, 
      });

      setEvaluationMetrics({
        test_accuracy: evalRes.data.test_accuracy || 0,
        fraud_detected_test: evalRes.data.fraud_detected_test || 0,
        total_fraud_present_test: evalRes.data.total_fraud_present_test || 0,
      });
      

      setShowResults(true);
    } catch (error) {
      console.error("Error analyzing file:", error);
      alert("Error processing file. Please try again.");
    }
  };
  const classificationData = classificationReport
  ? Object.entries(classificationReport).map(([label, metrics]) => ({
      class: label,
      Precision: metrics.precision,
      Recall: metrics.recall,
      F1Score: metrics["f1-score"],
    }))
  : [];
  
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Dashboard
        </h1>
      </div>

      <div className="grid grid-cols-1 gap-6">
        {/* Upload Form */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Transaction Analysis
          </h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label
                htmlFor="file-upload"
                className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
              >
                Upload Excel file
              </label>
              <input
                id="file-upload"
                type="file"
                accept=".xlsx,.xls,.csv"
                onChange={handleFileChange}
                className="block w-full text-sm text-gray-500 dark:text-gray-400
                         file:mr-4 file:py-2 file:px-4
                         file:rounded-md file:border-0
                         file:text-sm file:font-semibold
                         file:bg-blue-50 file:text-blue-700
                         dark:file:bg-blue-900/20 dark:file:text-blue-300
                         hover:file:bg-blue-100 dark:hover:file:bg-blue-800/30"
              />
            </div>
            <div className="flex items-center">
              <button
                type="submit"
                disabled={!selectedFile || isLoading}
                className="py-2 px-4 bg-blue-600 hover:bg-blue-700 
                         focus:ring-blue-500 focus:ring-offset-blue-200 
                         text-white w-full transition ease-in duration-200 
                         text-center text-base font-semibold shadow-md 
                         focus:outline-none focus:ring-2 focus:ring-offset-2 
                         rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? "Processing..." : "Analyze"}
              </button>
            </div>
            {selectedFile && (
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Selected: {selectedFile.name}
              </p>
            )}
          </form>
        </div>

        {/* Results Section */}
        {/* {showResults && (
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Random Forest Prediction
              </h2>
              <p className="text-gray-600 dark:text-gray-400">
                Fraud Transactions: {fraudCounts.rf}
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Logistic Regression Prediction
              </h2>
              <p className="text-gray-600 dark:text-gray-400">
                Fraud Transactions: {fraudCounts.lr}
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Neural Network Prediction
              </h2>
              <p className="text-gray-600 dark:text-gray-400">
                Fraud Transactions: {fraudCounts.nn}
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Logistic Regression (LOG)
              </h2>
              <p className="text-gray-600 dark:text-gray-400">
                Fraud Transactions: {fraudCounts.log}
              </p>
            </div>
          </div>
        )} */}
{/* {evaluationMetrics && evaluationMetrics.test_accuracy !== undefined && (
  <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
      Model Evaluation
    </h2>
    <p className="text-gray-600 dark:text-gray-400">
      <strong>Test Accuracy:</strong> {evaluationMetrics.test_accuracy?.toFixed(2)}
    </p>
    <p className="text-gray-600 dark:text-gray-400">
      <strong>Fraud Detected:</strong> {evaluationMetrics.fraud_detected_test} / {evaluationMetrics.total_fraud_present_test}
    </p>
  </div>
)} */}



 {classificationReport && confusionMatrix && (
  <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
      Classification Report
    </h2>

    <table className="min-w-full border-collapse border border-gray-300 dark:border-gray-600">
      <thead>
        <tr className="bg-gray-200 dark:bg-gray-700">
          <th className="border border-gray-300 dark:border-gray-600 p-2">Class</th>
          <th className="border border-gray-300 dark:border-gray-600 p-2">Precision</th>
          <th className="border border-gray-300 dark:border-gray-600 p-2">Recall</th>
          <th className="border border-gray-300 dark:border-gray-600 p-2">F1-Score</th>
          <th className="border border-gray-300 dark:border-gray-600 p-2">Support</th>
        </tr>
      </thead>
      <tbody>
        {["0", "1"].map((cls) => (
          <tr key={cls} className="text-center">
            <td className="border border-gray-300 dark:border-gray-600 p-2">{cls}</td>
            <td className="border border-gray-300 dark:border-gray-600 p-2">
              {classificationReport[cls].precision.toFixed(4)}
            </td>
            <td className="border border-gray-300 dark:border-gray-600 p-2">
              {classificationReport[cls].recall.toFixed(4)}
            </td>
            <td className="border border-gray-300 dark:border-gray-600 p-2">
              {classificationReport[cls]["f1-score"].toFixed(4)}
            </td>
            <td className="border border-gray-300 dark:border-gray-600 p-2">
              {classificationReport[cls].support}
            </td>
          </tr>
        ))}
        <tr className="text-center font-bold">
          <td className="border border-gray-300 dark:border-gray-600 p-2">Accuracy</td>
          <td className="border border-gray-300 dark:border-gray-600 p-2">-</td>
          <td className="border border-gray-300 dark:border-gray-600 p-2">-</td>
          <td className="border border-gray-300 dark:border-gray-600 p-2">-</td>
          <td className="border border-gray-300 dark:border-gray-600 p-2">
            {(classificationReport.accuracy * 100).toFixed(2)}%
          </td>
        </tr>
      </tbody>
    </table>

    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mt-6">
      Confusion Matrix
    </h2>
    <table className="min-w-full border-collapse border border-gray-300 dark:border-gray-600 mt-2">
      <thead>
        <tr className="bg-gray-200 dark:bg-gray-700 text-center">
          <th className="border border-gray-300 dark:border-gray-600 p-2">Actual \ Predicted</th>
          <th className="border border-gray-300 dark:border-gray-600 p-2">Class 0</th>
          <th className="border border-gray-300 dark:border-gray-600 p-2">Class 1</th>
        </tr>
      </thead>
      <tbody>
        <tr className="text-center">
          <td className="border border-gray-300 dark:border-gray-600 p-2">Class 0</td>
          <td className="border border-gray-300 dark:border-gray-600 p-2">{confusionMatrix[0][0]}</td>
          <td className="border border-gray-300 dark:border-gray-600 p-2">{confusionMatrix[0][1]}</td>
        </tr>
        <tr className="text-center">
          <td className="border border-gray-300 dark:border-gray-600 p-2">Class 1</td>
          <td className="border border-gray-300 dark:border-gray-600 p-2">{confusionMatrix[1][0]}</td>
          <td className="border border-gray-300 dark:border-gray-600 p-2">{confusionMatrix[1][1]}</td>
        </tr>
      </tbody>
    </table>
  </div>
)}


 {classificationReport1 && confusionMatrix1 && (

  
  <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
      Neural Network Classification Report
    </h2>

    <table className="min-w-full border-collapse border border-gray-300 dark:border-gray-600">
      <thead>
        <tr className="bg-gray-200 dark:bg-gray-700">
          <th className="border border-gray-300 dark:border-gray-600 p-2">Class</th>
          <th className="border border-gray-300 dark:border-gray-600 p-2">Precision</th>
          <th className="border border-gray-300 dark:border-gray-600 p-2">Recall</th>
          <th className="border border-gray-300 dark:border-gray-600 p-2">F1-Score</th>
          <th className="border border-gray-300 dark:border-gray-600 p-2">Support</th>
        </tr>
      </thead>
      <tbody>
        {["0", "1"].map((cls) => (
          <tr key={cls} className="text-center">
            <td className="border border-gray-300 dark:border-gray-600 p-2">{cls}</td>
            <td className="border border-gray-300 dark:border-gray-600 p-2">
              {classificationReport1[cls].precision.toFixed(4)}
            </td>
            <td className="border border-gray-300 dark:border-gray-600 p-2">
              {classificationReport1[cls].recall.toFixed(4)}
            </td>
            <td className="border border-gray-300 dark:border-gray-600 p-2">
              {classificationReport1[cls]["f1-score"].toFixed(4)}
            </td>
            <td className="border border-gray-300 dark:border-gray-600 p-2">
              {classificationReport1[cls].support}
            </td>
          </tr>
        ))}
        <tr className="text-center font-bold">
          <td className="border border-gray-300 dark:border-gray-600 p-2">Accuracy</td>
          <td className="border border-gray-300 dark:border-gray-600 p-2">-</td>
          <td className="border border-gray-300 dark:border-gray-600 p-2">-</td>
          <td className="border border-gray-300 dark:border-gray-600 p-2">-</td>
          <td className="border border-gray-300 dark:border-gray-600 p-2">
            {(accuracy * 100).toFixed(2)}%
          </td>
        </tr>
      </tbody>
    </table>

    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mt-6">
      Neural Network Confusion Matrix
    </h2>
    <table className="min-w-full border-collapse border border-gray-300 dark:border-gray-600 mt-2">
      <thead>
        <tr className="bg-gray-200 dark:bg-gray-700 text-center">
          <th className="border border-gray-300 dark:border-gray-600 p-2">Actual \ Predicted</th>
          <th className="border border-gray-300 dark:border-gray-600 p-2">Class 0</th>
          <th className="border border-gray-300 dark:border-gray-600 p-2">Class 1</th>
        </tr>
      </thead>
      <tbody>
        <tr className="text-center">
          <td className="border border-gray-300 dark:border-gray-600 p-2">Class 0</td>
          <td className="border border-gray-300 dark:border-gray-600 p-2">{confusionMatrix1[0][0]}</td>
          <td className="border border-gray-300 dark:border-gray-600 p-2">{confusionMatrix1[0][1]}</td>
        </tr>
        <tr className="text-center">
          <td className="border border-gray-300 dark:border-gray-600 p-2">Class 1</td>
          <td className="border border-gray-300 dark:border-gray-600 p-2">{confusionMatrix1[1][0]}</td>
          <td className="border border-gray-300 dark:border-gray-600 p-2">{confusionMatrix1[1][1]}</td>
        </tr>
      </tbody>
    </table>

    <h2 className="text-lg font-semibold text-gray-900 dark:text-white mt-6">
      Evaluation Metrics
    </h2>
    <p className="text-gray-800 dark:text-white">Accuracy: {accuracy?.toFixed(4)}</p>
    <p className="text-gray-800 dark:text-white">Recall: {recall?.toFixed(4)}</p>
    <p className="text-gray-800 dark:text-white">F1-Score: {f1Score?.toFixed(4)}</p>
  </div>
)}

      </div>
    </div>
  );
};

export default Dashboard;



