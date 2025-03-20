import React, { useState, useEffect } from "react";

const IntegratedFraudDocumentation = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [darkMode, setDarkMode] = useState(true); // Default to dark mode

  const tabs = [
    { name: "Getting Started" },
    { name: "Dashboard Guide" },
    { name: "Alert System" },
    { name: "API Reference" },
  ];

  // Optional: Detect system preference for dark mode
  useEffect(() => {
    // Check if user has a system preference
    const prefersDarkMode = window.matchMedia(
      "(prefers-color-scheme: dark)"
    ).matches;
    setDarkMode(prefersDarkMode);
  }, []);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  return (
    <div
      className={`max-w-6xl mx-auto p-6 rounded-lg ${
        darkMode ? "bg-gray-900 text-gray-100" : "bg-white text-gray-900"
      } shadow-md transition-colors duration-200`}
    >
      {/* Mode Toggle */}
      <div className="flex justify-end mb-4">
        <button
          onClick={toggleDarkMode}
          className={`px-4 py-2 rounded-md text-sm font-medium ${
            darkMode ? "bg-gray-700 text-gray-100" : "bg-gray-200 text-gray-800"
          }`}
        >
          {darkMode ? "Light Mode" : "Dark Mode"}
        </button>
      </div>

      {/* Documentation Header */}
      <div
        className={`border-b ${
          darkMode ? "border-gray-700" : "border-gray-200"
        } pb-4 mb-6`}
      >
        <h1 className="text-2xl font-bold">
          Fraud Detection System Documentation
        </h1>
        <p className={`mt-1 ${darkMode ? "text-gray-400" : "text-gray-600"}`}>
          Complete guide for using the real-time fraud detection platform
        </p>
      </div>

      {/* Tabs Navigation */}
      <div
        className={`border-b ${
          darkMode ? "border-gray-700" : "border-gray-200"
        } mb-6`}
      >
        <div className="flex">
          {tabs.map((tab, index) => (
            <button
              key={index}
              className={`px-6 py-3 font-medium text-sm focus:outline-none ${
                activeTab === index
                  ? darkMode
                    ? "border-b-2 border-blue-400 text-blue-400"
                    : "border-b-2 border-blue-500 text-blue-600"
                  : darkMode
                  ? "text-gray-400 hover:text-gray-200"
                  : "text-gray-500 hover:text-gray-700"
              }`}
              onClick={() => setActiveTab(index)}
            >
              {tab.name}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div className="mb-8">
        {/* Getting Started Content */}
        <div className={activeTab === 0 ? "block" : "hidden"}>
          <h2 className="text-xl font-semibold mb-4">
            Getting Started with Fraud Detection
          </h2>
          <p className="mb-6">
            Welcome to our fraud detection platform documentation. This guide
            will help you set up and use our system effectively to monitor and
            prevent fraudulent transactions.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
            {/* Quick Start Card */}
            <div
              className={`rounded-lg shadow-md p-6 border ${
                darkMode
                  ? "bg-gray-800 border-gray-700"
                  : "bg-white border-gray-100"
              }`}
            >
              <div className="flex justify-between items-center mb-4">
                <h3 className="font-bold text-lg">Quick Start Guide</h3>
                <span
                  className={`px-2 py-1 rounded text-xs font-medium ${
                    darkMode
                      ? "bg-blue-900 text-blue-300"
                      : "bg-blue-100 text-blue-800"
                  }`}
                >
                  Beginner
                </span>
              </div>
              <p
                className={`mb-4 ${
                  darkMode ? "text-gray-300" : "text-gray-700"
                }`}
              >
                Follow these steps to quickly set up your fraud detection
                dashboard and start monitoring transactions.
              </p>
              <ul
                className={`list-disc pl-5 space-y-2 ${
                  darkMode ? "text-gray-300" : "text-gray-700"
                }`}
              >
                <li>Set up user authentication</li>
                <li>Configure alert thresholds</li>
                <li>Connect payment gateways</li>
                <li>Review default detection rules</li>
              </ul>
              <button
                className={`mt-4 font-medium ${
                  darkMode
                    ? "text-blue-400 hover:text-blue-300"
                    : "text-blue-600 hover:text-blue-800"
                }`}
              >
                Read full guide →
              </button>
            </div>

            {/* System Requirements Card */}
            <div
              className={`rounded-lg shadow-md p-6 border ${
                darkMode
                  ? "bg-gray-800 border-gray-700"
                  : "bg-white border-gray-100"
              }`}
            >
              <div className="flex justify-between items-center mb-4">
                <h3 className="font-bold text-lg">System Requirements</h3>
                <span
                  className={`px-2 py-1 rounded text-xs font-medium ${
                    darkMode
                      ? "bg-gray-700 text-gray-300"
                      : "bg-gray-100 text-gray-800"
                  }`}
                >
                  Technical
                </span>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span
                    className={darkMode ? "text-gray-400" : "text-gray-500"}
                  >
                    Supported Browsers:
                  </span>
                  <span>Chrome, Firefox, Safari, Edge</span>
                </div>
                <div className="flex justify-between">
                  <span
                    className={darkMode ? "text-gray-400" : "text-gray-500"}
                  >
                    API Version:
                  </span>
                  <span className="font-mono">v2.1.4+</span>
                </div>
                <div className="flex justify-between">
                  <span
                    className={darkMode ? "text-gray-400" : "text-gray-500"}
                  >
                    Data Retention:
                  </span>
                  <span>90 days minimum</span>
                </div>
                <div className="flex justify-between">
                  <span
                    className={darkMode ? "text-gray-400" : "text-gray-500"}
                  >
                    Authentication:
                  </span>
                  <span>OAuth 2.0, API Keys</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Dashboard Guide Content */}
        <div className={activeTab === 1 ? "block" : "hidden"}>
          <h2 className="text-xl font-semibold mb-4">Dashboard Guide</h2>
          <p className="mb-6">
            Learn how to interpret and use the fraud detection dashboard to
            monitor transaction activity and respond to potential threats.
          </p>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
            {/* Dashboard Overview Card */}
            <div
              className={`rounded-lg shadow-md p-6 border ${
                darkMode
                  ? "bg-gray-800 border-gray-700"
                  : "bg-white border-gray-100"
              }`}
            >
              <h3 className="font-bold text-lg mb-3">Dashboard Elements</h3>
              <p
                className={`mb-4 ${
                  darkMode ? "text-gray-300" : "text-gray-700"
                }`}
              >
                Understanding the key components of your fraud monitoring
                dashboard.
              </p>
              <ul
                className={`list-disc pl-5 space-y-1 ${
                  darkMode ? "text-gray-300" : "text-gray-700"
                }`}
              >
                <li>Risk score gauge</li>
                <li>Transaction timeline</li>
                <li>Geographic heat map</li>
                <li>Alert notification panel</li>
              </ul>
            </div>

            {/* Custom Metrics Card */}
            <div
              className={`rounded-lg shadow-md p-6 border ${
                darkMode
                  ? "bg-gray-800 border-gray-700"
                  : "bg-white border-gray-100"
              }`}
            >
              <h3 className="font-bold text-lg mb-3">Custom Metrics</h3>
              <p
                className={`mb-4 ${
                  darkMode ? "text-gray-300" : "text-gray-700"
                }`}
              >
                Create and customize metrics to monitor specific fraud patterns
                relevant to your business.
              </p>
              <div
                className={`p-3 rounded text-sm ${
                  darkMode ? "bg-gray-700" : "bg-gray-50"
                }`}
              >
                <code>
                  Tip: Use the metric builder to combine multiple risk factors
                  into a single score.
                </code>
              </div>
            </div>

            {/* Alert Configuration Card */}
            <div
              className={`rounded-lg shadow-md p-6 border ${
                darkMode
                  ? "bg-gray-800 border-gray-700"
                  : "bg-white border-gray-100"
              }`}
            >
              <h3 className="font-bold text-lg mb-3">Alert Configuration</h3>
              <p
                className={`mb-4 ${
                  darkMode ? "text-gray-300" : "text-gray-700"
                }`}
              >
                Set up automated alerts based on suspicious activity thresholds.
              </p>
              <div
                className={`border-l-4 p-3 ${
                  darkMode
                    ? "bg-yellow-900 border-yellow-700 text-yellow-300"
                    : "bg-yellow-50 border-yellow-400 text-yellow-700"
                }`}
              >
                <p className="text-sm">
                  Configure notification channels in the Settings section before
                  creating new alerts.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Alert System Content */}
        <div className={activeTab === 2 ? "block" : "hidden"}>
          <h2 className="text-xl font-semibold mb-4">Alert System Guide</h2>
          <p className="mb-6">
            Learn how to configure and respond to fraud alerts generated by the
            system.
          </p>

          <div
            className={`mb-6 rounded-lg shadow-md p-6 border ${
              darkMode
                ? "bg-gray-800 border-gray-700"
                : "bg-white border-gray-100"
            }`}
          >
            <div className="flex justify-between items-center mb-4">
              <h3 className="font-bold text-lg">
                Sample Alert: Unusual Location
              </h3>
              <span
                className={`px-2 py-1 rounded text-xs font-medium ${
                  darkMode
                    ? "bg-red-900 text-red-300"
                    : "bg-red-100 text-red-800"
                }`}
              >
                High Risk
              </span>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between">
                <span className={darkMode ? "text-gray-400" : "text-gray-500"}>
                  Alert Type:
                </span>
                <span>Geolocation Mismatch</span>
              </div>
              <div className="flex justify-between">
                <span className={darkMode ? "text-gray-400" : "text-gray-500"}>
                  Trigger:
                </span>
                <span>Transaction from new country</span>
              </div>
              <div className="flex justify-between">
                <span className={darkMode ? "text-gray-400" : "text-gray-500"}>
                  Recommended Action:
                </span>
                <span
                  className={`font-medium ${
                    darkMode ? "text-red-400" : "text-red-600"
                  }`}
                >
                  Block and verify
                </span>
              </div>
            </div>

            <div
              className={`mt-5 pt-4 border-t ${
                darkMode ? "border-gray-700" : "border-gray-100"
              }`}
            >
              <p
                className={`text-sm ${
                  darkMode ? "text-gray-300" : "text-gray-700"
                }`}
              >
                This alert triggers when a transaction originates from a country
                where the customer has not previously transacted. Consider
                implementing step-up authentication for these cases.
              </p>
            </div>
          </div>

          <div
            className={`rounded-lg shadow-md p-6 border ${
              darkMode
                ? "bg-gray-800 border-gray-700"
                : "bg-white border-gray-100"
            }`}
          >
            <h3 className="font-bold text-lg mb-3">Alert Response Protocol</h3>
            <p
              className={`mb-4 ${darkMode ? "text-gray-300" : "text-gray-700"}`}
            >
              Follow these steps when responding to fraud alerts:
            </p>
            <ol
              className={`list-decimal pl-5 space-y-2 ${
                darkMode ? "text-gray-300" : "text-gray-700"
              }`}
            >
              <li>Review alert details and risk factors</li>
              <li>Check customer transaction history</li>
              <li>Determine appropriate action (approve, review, block)</li>
              <li>Document decision rationale</li>
              <li>Update rules if patterns emerge</li>
            </ol>
          </div>
        </div>

        {/* API Reference Content */}
        <div className={activeTab === 3 ? "block" : "hidden"}>
          <h2 className="text-xl font-semibold mb-4">API Reference</h2>
          <p className="mb-6">
            Technical documentation for integrating with the fraud detection
            API.
          </p>

          <div
            className={`mb-6 rounded-lg shadow-md p-6 border ${
              darkMode
                ? "bg-gray-800 border-gray-700"
                : "bg-white border-gray-100"
            }`}
          >
            <h3 className="font-bold text-lg mb-3">
              Transaction Verification Endpoint
            </h3>
            <div
              className={`p-4 rounded-md font-mono text-sm mb-4 ${
                darkMode ? "bg-gray-700" : "bg-gray-50"
              }`}
            >
              POST /api/v2/transactions/verify
            </div>

            <h4 className="font-semibold mt-4 mb-2">Request Parameters</h4>
            <div className="overflow-x-auto">
              <table
                className={`min-w-full ${
                  darkMode ? "bg-gray-800" : "bg-white"
                }`}
              >
                <thead>
                  <tr className={darkMode ? "bg-gray-700" : "bg-gray-50"}>
                    <th
                      className={`py-2 px-4 text-left ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      Parameter
                    </th>
                    <th
                      className={`py-2 px-4 text-left ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      Type
                    </th>
                    <th
                      className={`py-2 px-4 text-left ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      Required
                    </th>
                    <th
                      className={`py-2 px-4 text-left ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      Description
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td
                      className={`py-2 px-4 font-mono ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      transaction_id
                    </td>
                    <td
                      className={`py-2 px-4 ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      String
                    </td>
                    <td
                      className={`py-2 px-4 ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      Yes
                    </td>
                    <td
                      className={`py-2 px-4 ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      Unique transaction identifier
                    </td>
                  </tr>
                  <tr>
                    <td
                      className={`py-2 px-4 font-mono ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      amount
                    </td>
                    <td
                      className={`py-2 px-4 ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      Number
                    </td>
                    <td
                      className={`py-2 px-4 ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      Yes
                    </td>
                    <td
                      className={`py-2 px-4 ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      Transaction amount
                    </td>
                  </tr>
                  <tr>
                    <td
                      className={`py-2 px-4 font-mono ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      ip_address
                    </td>
                    <td
                      className={`py-2 px-4 ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      String
                    </td>
                    <td
                      className={`py-2 px-4 ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      Yes
                    </td>
                    <td
                      className={`py-2 px-4 ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      IP address of the customer
                    </td>
                  </tr>
                  <tr>
                    <td
                      className={`py-2 px-4 font-mono ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      user_id
                    </td>
                    <td
                      className={`py-2 px-4 ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      String
                    </td>
                    <td
                      className={`py-2 px-4 ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      Yes
                    </td>
                    <td
                      className={`py-2 px-4 ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      Customer identifier
                    </td>
                  </tr>
                  <tr>
                    <td
                      className={`py-2 px-4 font-mono ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      timestamp
                    </td>
                    <td
                      className={`py-2 px-4 ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      String
                    </td>
                    <td
                      className={`py-2 px-4 ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      Yes
                    </td>
                    <td
                      className={`py-2 px-4 ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      ISO 8601 formatted date-time
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <h4 className="font-semibold mt-6 mb-2">Response Format</h4>
            <div
              className={`p-4 rounded-md font-mono text-sm mb-4 whitespace-pre ${
                darkMode
                  ? "bg-gray-700 text-gray-200"
                  : "bg-gray-50 text-gray-800"
              }`}
            >
              {`{
  "status": "success",
  "risk_score": 27,
  "recommendation": "approve",
  "risk_factors": [
    {
      "type": "velocity",
      "score": 15,
      "description": "Multiple transactions in short period"
    }
  ],
  "transaction_id": "t_12345abcde"
}`}
            </div>

            <h4 className="font-semibold mt-6 mb-2">Error Codes</h4>
            <div className="overflow-x-auto">
              <table
                className={`min-w-full ${
                  darkMode ? "bg-gray-800" : "bg-white"
                }`}
              >
                <thead>
                  <tr className={darkMode ? "bg-gray-700" : "bg-gray-50"}>
                    <th
                      className={`py-2 px-4 text-left ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      Code
                    </th>
                    <th
                      className={`py-2 px-4 text-left ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      Description
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td
                      className={`py-2 px-4 font-mono ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      400
                    </td>
                    <td
                      className={`py-2 px-4 ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      Missing required parameters
                    </td>
                  </tr>
                  <tr>
                    <td
                      className={`py-2 px-4 font-mono ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      401
                    </td>
                    <td
                      className={`py-2 px-4 ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      Invalid API key
                    </td>
                  </tr>
                  <tr>
                    <td
                      className={`py-2 px-4 font-mono ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      429
                    </td>
                    <td
                      className={`py-2 px-4 ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      Rate limit exceeded
                    </td>
                  </tr>
                  <tr>
                    <td
                      className={`py-2 px-4 font-mono ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      500
                    </td>
                    <td
                      className={`py-2 px-4 ${
                        darkMode ? "border-b border-gray-600" : "border-b"
                      }`}
                    >
                      Internal server error
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div
            className={`rounded-lg shadow-md p-6 border ${
              darkMode
                ? "bg-gray-800 border-gray-700"
                : "bg-white border-gray-100"
            }`}
          >
            <h3 className="font-bold text-lg mb-3">Authentication</h3>
            <p
              className={`mb-4 ${darkMode ? "text-gray-300" : "text-gray-700"}`}
            >
              All API requests require authentication using an API key.
            </p>

            <div
              className={`p-4 rounded-md font-mono text-sm mb-4 ${
                darkMode ? "bg-gray-700" : "bg-gray-50"
              }`}
            >
              Authorization: Bearer YOUR_API_KEY
            </div>

            <div
              className={`border-l-4 p-3 ${
                darkMode
                  ? "bg-blue-900 border-blue-700 text-blue-300"
                  : "bg-blue-50 border-blue-400 text-blue-700"
              }`}
            >
              <p className="text-sm">
                API keys can be generated and managed in the Account Settings
                section. Keep your API keys secure and never share them in
                client-side code.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div
        className={`pt-4 border-t ${
          darkMode ? "border-gray-700" : "border-gray-200"
        }`}
      >
        <div className="flex flex-col md:flex-row md:justify-between">
          <div className="mb-4 md:mb-0">
            <h4 className="font-semibold">Need more help?</h4>
            <p
              className={`mt-1 ${darkMode ? "text-gray-400" : "text-gray-600"}`}
            >
              Contact our support team for assistance.
            </p>
          </div>
          <div>
            <button
              className={`px-4 py-2 rounded font-medium ${
                darkMode
                  ? "bg-blue-600 hover:bg-blue-700 text-white"
                  : "bg-blue-500 hover:bg-blue-600 text-white"
              }`}
            >
              Contact Support
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default IntegratedFraudDocumentation;
