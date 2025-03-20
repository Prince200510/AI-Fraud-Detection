import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import { Provider } from "react-redux";
import { store } from "./store";
import Login from "./pages/Login";
import Layout from "./components/layout/Layout";
import Dashboard from "./pages/Dashboard";
import Documentation from "./pages/Documentation";
import PrivateRoute from "./components/PrivateRoute";
import Aichat from "./pages/Ai-chat";

function App() {
  return (
    <Provider store={store}>
      <Router>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route
            path="/"
            element={
              <PrivateRoute>
                <Layout />
              </PrivateRoute>
            }
          >
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="docs" element={<Documentation />} />
            <Route path="ai-chat" element={<Aichat />} />
            {/* <Route path="*" element={<Navigate to="/dashboard" replace />} /> */}
            <Route path="Logout" element={<Navigate to="/Login" />} />
          </Route>
        </Routes>
      </Router>
    </Provider>
  );
}

export default App;
