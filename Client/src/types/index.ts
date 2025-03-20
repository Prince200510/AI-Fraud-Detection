export interface User {
  username: string;
  personalId: string;
}

export interface AuthState {
  user: User | null;
  isLoading: boolean;
  error: string | null;
  isAuthenticated: boolean;
}

export interface AnalysisResult {
  transactionScore: number;
  riskLevel: "low" | "medium" | "high";
  anomalyCount: number;
  aiRecommendations: string[];
  confidenceScore: number;
}

export interface SidebarItem {
  icon: React.ComponentType;
  label: string;
  path: string;
}

export interface FileUploadState {
  file: File | null;
  progress: number;
  error: string | null;
  isUploading: boolean;
}

export interface DashboardState {
  analysisResult: AnalysisResult | null;
  isLoading: boolean;
  error: string | null;
}
