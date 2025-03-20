import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import type { DashboardState, AnalysisResult } from '../../types';

const initialState: DashboardState = {
  analysisResult: null,
  isLoading: false,
  error: null,
};

const dashboardSlice = createSlice({
  name: 'dashboard',
  initialState,
  reducers: {
    analysisStart: (state) => {
      state.isLoading = true;
      state.error = null;
    },
    analysisSuccess: (state, action: PayloadAction<AnalysisResult>) => {
      state.isLoading = false;
      state.analysisResult = action.payload;
      state.error = null;
    },
    analysisFailure: (state, action: PayloadAction<string>) => {
      state.isLoading = false;
      state.error = action.payload;
    },
    resetAnalysis: (state) => {
      state.analysisResult = null;
      state.error = null;
      state.isLoading = false;
    },
  },
});

export const {
  analysisStart,
  analysisSuccess,
  analysisFailure,
  resetAnalysis,
} = dashboardSlice.actions;

export default dashboardSlice.reducer;