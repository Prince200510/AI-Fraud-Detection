import { useSelector } from "react-redux";
import { Shield, Bell, User } from "lucide-react";
import type { RootState } from "../../store";
import ThemeToggle from "../ThemeToggle";

const Header = () => {
  const user = useSelector((state: RootState) => state.auth.user);

  return (
    <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
      <div className="h-16 px-4 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Shield className="h-8 w-8 text-blue-600 dark:text-blue-400" />
          <h1 className="text-xl font-bold text-gray-900 dark:text-white">
            FraudEye
          </h1>
        </div>

        <div className="flex items-center space-x-4 pr-10">
          <ThemeToggle />
          <button
            className="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
            aria-label="Notifications"
          >
            <Bell className="h-5 w-5 text-gray-500 dark:text-gray-400" />
          </button>
          <div className="flex items-center space-x-2">
            <User className="h-5 w-5 text-gray-500 dark:text-gray-400 " />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-500">
              {user?.username}
            </span>
            <button
              className="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
              aria-label="User menu"
            ></button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
