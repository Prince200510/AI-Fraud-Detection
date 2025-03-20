import React, { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { fetchFraudulentTransactions } from "../redux/actions/fraudActions";
import { RootState } from "../redux/store";
import { Card, CardContent } from "@/components/ui/card";
import { Table, TableHead, TableRow, TableCell, TableBody } from "@/components/ui/table";
import { Loader } from "@/components/ui/loader";
import { Alert } from "@/components/ui/alert";

const FraudDetection: React.FC = () => {
  const dispatch = useDispatch();
  const { fraudTransactions, loading, error } = useSelector(
    (state: RootState) => state.fraud
  );

  useEffect(() => {
    dispatch(fetchFraudulentTransactions());
  }, [dispatch]);

  return (
    <Card className="p-4 shadow-lg">
      <CardContent>
        <h2 className="text-xl font-semibold mb-4">Fraudulent Transactions</h2>
        {loading ? (
          <Loader />
        ) : error ? (
          <Alert variant="destructive">{error}</Alert>
        ) : (
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>ID</TableCell>
                <TableCell>Amount</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Risk Level</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {fraudTransactions.map((tx) => (
                <TableRow key={tx.id}>
                  <TableCell>{tx.id}</TableCell>
                  <TableCell>${tx.amount.toFixed(2)}</TableCell>
                  <TableCell>{tx.status}</TableCell>
                  <TableCell className={
                    tx.riskLevel === "high" ? "text-red-500" : "text-yellow-500"
                  }>
                    {tx.riskLevel.toUpperCase()}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
};

export default FraudDetection;


// {classificationReport && confusionMatrix && (
//   <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
//     <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
//       Classification Report
//     </h2>

//     <table className="min-w-full border-collapse border border-gray-300 dark:border-gray-600">
//       <thead>
//         <tr className="bg-gray-200 dark:bg-gray-700">
//           <th className="border border-gray-300 dark:border-gray-600 p-2">Class</th>
//           <th className="border border-gray-300 dark:border-gray-600 p-2">Precision</th>
//           <th className="border border-gray-300 dark:border-gray-600 p-2">Recall</th>
//           <th className="border border-gray-300 dark:border-gray-600 p-2">F1-Score</th>
//           <th className="border border-gray-300 dark:border-gray-600 p-2">Support</th>
//         </tr>
//       </thead>
//       <tbody>
//         {["0", "1"].map((cls) => (
//           <tr key={cls} className="text-center">
//             <td className="border border-gray-300 dark:border-gray-600 p-2">{cls}</td>
//             <td className="border border-gray-300 dark:border-gray-600 p-2">
//               {classificationReport[cls].precision.toFixed(4)}
//             </td>
//             <td className="border border-gray-300 dark:border-gray-600 p-2">
//               {classificationReport[cls].recall.toFixed(4)}
//             </td>
//             <td className="border border-gray-300 dark:border-gray-600 p-2">
//               {classificationReport[cls]["f1-score"].toFixed(4)}
//             </td>
//             <td className="border border-gray-300 dark:border-gray-600 p-2">
//               {classificationReport[cls].support}
//             </td>
//           </tr>
//         ))}
//         <tr className="text-center font-bold">
//           <td className="border border-gray-300 dark:border-gray-600 p-2">Accuracy</td>
//           <td className="border border-gray-300 dark:border-gray-600 p-2">-</td>
//           <td className="border border-gray-300 dark:border-gray-600 p-2">-</td>
//           <td className="border border-gray-300 dark:border-gray-600 p-2">-</td>
//           <td className="border border-gray-300 dark:border-gray-600 p-2">
//             {(classificationReport.accuracy * 100).toFixed(2)}%
//           </td>
//         </tr>
//       </tbody>
//     </table>

//     <h2 className="text-lg font-semibold text-gray-900 dark:text-white mt-6">
//       Confusion Matrix
//     </h2>
//     <table className="min-w-full border-collapse border border-gray-300 dark:border-gray-600 mt-2">
//       <thead>
//         <tr className="bg-gray-200 dark:bg-gray-700 text-center">
//           <th className="border border-gray-300 dark:border-gray-600 p-2">Actual \ Predicted</th>
//           <th className="border border-gray-300 dark:border-gray-600 p-2">Class 0</th>
//           <th className="border border-gray-300 dark:border-gray-600 p-2">Class 1</th>
//         </tr>
//       </thead>
//       <tbody>
//         <tr className="text-center">
//           <td className="border border-gray-300 dark:border-gray-600 p-2">Class 0</td>
//           <td className="border border-gray-300 dark:border-gray-600 p-2">{confusionMatrix[0][0]}</td>
//           <td className="border border-gray-300 dark:border-gray-600 p-2">{confusionMatrix[0][1]}</td>
//         </tr>
//         <tr className="text-center">
//           <td className="border border-gray-300 dark:border-gray-600 p-2">Class 1</td>
//           <td className="border border-gray-300 dark:border-gray-600 p-2">{confusionMatrix[1][0]}</td>
//           <td className="border border-gray-300 dark:border-gray-600 p-2">{confusionMatrix[1][1]}</td>
//         </tr>
//       </tbody>
//     </table>
//   </div>
// )}
