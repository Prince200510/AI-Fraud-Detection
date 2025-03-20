import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import type { AuthState, User } from "../../types";

const initialState: AuthState = {
  user: null,
  isLoading: false,
  error: null,
  isAuthenticated: false,
};

const authSlice = createSlice({
  name: "auth",
  initialState,
  reducers: {
    loginStart: (state) => {
      state.isLoading = true;
      state.error = null;
    },
    loginSuccess: (state, action: PayloadAction<User>) => {
      state.isLoading = false;
      state.user = action.payload;
      state.error = null;
    },
    loginFailure: (state, action: PayloadAction<string>) => {
      state.isLoading = false;
      state.error = action.payload;
    },
    logout: (state) => {
      state.user = null;
      state.error = null;
    },
  },
  extraReducers: () => {
    // You can add any cross-slice logic here if needed
  },
});

export const { loginStart, loginSuccess, loginFailure, logout } =
  authSlice.actions;

// Selector for checking authentication status
export const selectIsAuthenticated = (state: { auth: AuthState }) =>
  state.auth.user !== null;

export default authSlice.reducer;
