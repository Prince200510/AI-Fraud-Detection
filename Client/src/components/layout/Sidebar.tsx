import { useState } from "react";
import { NavLink } from "react-router-dom";
import {
  ChevronLeft,
  Home,
  FileText,
  HelpCircle,
  Menu,
  BotMessageSquare,
  LogOut,
} from "lucide-react";
import { cn } from "../../utils/cn";
import type { SidebarItem } from "../../types";

const sidebarItems: SidebarItem[] = [
  { icon: Home, label: "Dashboard", path: "/dashboard" },
  { icon: FileText, label: "Documentation", path: "/docs" },
  { icon: BotMessageSquare, label: "AI Chat", path: "/Ai-chat" },
  // { icon: HelpCircle, label: "Help", path: "/help" },
  { icon: LogOut, label: "Logout", path: "/Logout" },
];

const Sidebar = () => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  return (
    <div
      className={cn(
        "flex flex-col h-screen bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 transition-all duration-300",
        isCollapsed ? "w-16" : "w-64"
      )}
    >
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
        {!isCollapsed && (
          <span className="text-xl font-bold text-gray-800 dark:text-white">
            FraudEye
          </span>
        )}
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
          aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {isCollapsed ? (
            <Menu className="h-5 w-5 text-gray-500 dark:text-gray-400" />
          ) : (
            <ChevronLeft className="h-5 w-5 text-gray-500 dark:text-gray-400" />
          )}
        </button>
      </div>

      <nav className="flex-1 overflow-y-auto p-4">
        <ul className="space-y-2">
          {sidebarItems.map((item) => (
            <li key={item.path}>
              <NavLink
                to={item.path}
                className={({ isActive }) =>
                  cn(
                    "flex items-center space-x-3 p-2.5 rounded-lg transition-colors",
                    isActive
                      ? "bg-blue-50 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400"
                      : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                  )
                }
              >
                <item.icon
                  className={cn("h-5 w-5", isCollapsed ? "mx-auto" : "")}
                />
                {!isCollapsed && <span>{item.label}</span>}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
    </div>
  );
};

export default Sidebar;
