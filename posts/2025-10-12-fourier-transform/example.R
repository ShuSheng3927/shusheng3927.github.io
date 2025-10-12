# ------------------------------------------------
# Libraries
# ------------------------------------------------
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# ------------------------------------------------
# Parameters
# ------------------------------------------------
fs  <- 10      # sampling frequency [Hz]
T   <- 2       # duration [s]
N   <- fs * T
t   <- seq(0, T - 1/fs, by = 1/fs)  # discrete sample times

# Two-tone signal
f1 <- 3.0
f2 <- 0.8
x  <- sin(2*pi*f1*t) + 3*sin(2*pi*f2*t)

# High-res "continuous" reference
N_cont  <- 5000
t_cont  <- seq(0, T, length.out = N_cont)
x1_cont <- sin(2*pi*f1*t_cont)
x2_cont <- 3*sin(2*pi*f2*t_cont)
x_cont  <- x1_cont + x2_cont

# ------------------------------------------------
# FFT (DFT) -> two-sided spectrum with fftshift-style ordering
# ------------------------------------------------
X <- fft(x)                # unnormalized in R
mag_full <- Mod(X) / N     # two-sided magnitude (no doubling)

# Reorder so frequency runs from -fs/2 ... +fs/2 (even N here)
idx_shift  <- c((N/2 + 1):N, 1:(N/2))
freq_shift <- seq(-fs/2, fs/2 - fs/N, by = fs/N)  # length N
mag_shift  <- mag_full[idx_shift]

df_spec2 <- tibble(freq = freq_shift, amp = mag_shift)

# Component markers (±f1, ±f2)
df_lines <- tibble(
  freq  = c(+f1, -f1, +f2, -f2),
  label = rep(c("3 Hz", "0.8 Hz"), each = 2)
)

# ------------------------------------------------
# Bandlimited Reconstruction via Zero-Padded DFT
# ------------------------------------------------
N_up <- N_cont
X_up <- complex(real = rep(0, N_up), imaginary = rep(0, N_up))

if (N %% 2 == 0) {
  # Copy DC..Nyquist
  X_up[1:(N/2 + 1)] <- X[1:(N/2 + 1)]
  # Copy negative freqs (exclude DC & Nyquist)
  if (N/2 - 1 >= 1) {
    X_up[(N_up - (N/2 - 1) + 1):N_up] <- X[(N/2 + 2):N]
  }
} else {
  half <- floor(N/2)
  X_up[1:(half + 1)] <- X[1:(half + 1)]
  if (half >= 1) {
    X_up[(N_up - half + 1):N_up] <- X[(half + 2):N]
  }
}
# Amplitude correction when changing IFFT length
X_up <- (N_up / N) * X_up
x_rec <- Re(fft(X_up, inverse = TRUE)) / N_up
t_rec <- seq(0, T, length.out = N_up)

# ------------------------------------------------
# Data frames for plotting
# ------------------------------------------------
# Top: components + total + samples
df_cont <- tibble(
  t = t_cont,
  `3 Hz`     = x1_cont,
  `0.8 Hz`   = x2_cont,
  `Total`    = x_cont
) |>
  pivot_longer(-t, names_to = "series", values_to = "value")

df_samples <- tibble(t = t, Samples = x)

# Bottom: reconstruction vs ground truth
df_truth <- tibble(t = t_cont, `Ground Truth` = x_cont)
df_recon <- tibble(t = t_rec,  `Reconstruction` = x_rec)

# ------------------------------------------------
# Palette
# ------------------------------------------------
pal <- c(
  "3 Hz"           = "#1f77b4",
  "0.8 Hz"         = "#2ca02c",
  "Total"          = "#d62728",
  "Samples"        = "#000000",
  "Reconstruction" = "#9467bd",
  "Ground Truth"   = "#d62728",
  "Spectrum"       = "#7f7f7f"
)

# ------------------------------------------------
# Plot 1 (top): components + total + samples (point-only for Samples)
# ------------------------------------------------
top_breaks <- c("3 Hz", "0.8 Hz", "Total", "Samples")

p_top <- ggplot() +
  geom_line(
    data = df_cont |> filter(series %in% c("3 Hz", "0.8 Hz")),
    aes(t, value, color = series),
    linewidth = 0.6, alpha = 0.95, lty="dashed"
  ) +
  geom_line(
    data = df_cont |> filter(series == "Total"),
    aes(t, value, color = series),
    linewidth = 0.8, alpha = 0.9
  ) +
  geom_point(
    data = df_samples,
    aes(t, Samples, color = "Samples"),
    size = 2, alpha = 0.9, show.legend = TRUE
  ) +
  scale_color_manual(values = pal, breaks = top_breaks) +
  guides(color = guide_legend(
    override.aes = list(
      linetype = c(2, 2, 1, 0),
      shape    = c(NA, NA, NA, 16),
      linewidth= c(0.6, 0.6, 0.8, NA)
    )
  )) +
  labs(
    title = "Signals: Components, Total, and Samples",
    x = "Time (s)", y = "Amplitude", color = NULL
  ) +
  theme_minimal(base_size = 12)

# ------------------------------------------------
# Plot 2 (middle): TWO-SIDED spectrum as stems; simple legend with horizontal keys
# ------------------------------------------------
p_mid <- ggplot() +
  ylim(0,1) + 
  xlim(-5,5) + 
  # Spectrum stems (no legend)
  geom_segment(
    data = df_spec2,
    aes(x = freq, xend = freq, y = 0, yend = amp),
    color = pal["Spectrum"],
    linewidth = 0.9,
    show.legend = FALSE
  ) +
  # Component frequency markers (vertical lines, no legend)
  geom_segment(
    data = df_lines,
    aes(x = freq, xend = freq, y = 0, yend = 1, color = label, linetype = label),
    linewidth = 0.6,
    show.legend = FALSE
  ) +
  scale_color_manual(
    values = pal[c("Spectrum","3 Hz","0.8 Hz")],
    breaks = c("Spectrum","3 Hz","0.8 Hz"),
    name   = NULL
  ) +
  scale_linetype_manual(
    values = c("Spectrum"="solid","3 Hz"="dashed","0.8 Hz"="dashed"),
    breaks = c("Spectrum","3 Hz","0.8 Hz"),
    name   = NULL
  ) +
    # Dummy HORIZONTAL lines to generate legend
  geom_hline(
    aes(yintercept = -1, color = "Spectrum", linetype = "Spectrum"),
    linewidth = 0.6,
    show.legend = TRUE
  ) +
  geom_hline(
    aes(yintercept = -1, color = "3 Hz", linetype = "3 Hz"),
    linewidth = 0.6,
    show.legend = TRUE
  ) +
  geom_hline(
    aes(yintercept = -1, color = "0.8 Hz", linetype = "0.8 Hz"),
    linewidth = 0.9,
    show.legend = TRUE
  ) +
  labs(
    title = "Two-Sided Amplitude Spectrum from DFT",
    x = "Frequency (Hz)", y = "Amplitude"
  ) +
  theme_minimal(base_size = 12)


# ------------------------------------------------
# Plot 3 (bottom): reconstruction vs ground truth
# ------------------------------------------------
p_bot <- ggplot() +
  geom_line(
    data = df_truth,
    aes(t, `Ground Truth`, color = "Ground Truth"),
    linewidth = 0.9
  ) +
  geom_line(
    data = df_recon,
    aes(t, `Reconstruction`, color = "Reconstruction"),
    linewidth = 0.9, alpha = 0.9
  ) +
  scale_color_manual(values = pal[c("Ground Truth","Reconstruction")]) +
  labs(
    title = "Signal Reconstruction from DFT vs Ground Truth",
    x = "Time (s)", y = "Amplitude", color = NULL
  ) +
  theme_minimal(base_size = 12)

# ------------------------------------------------
# Arrange: three rows
# ------------------------------------------------

# ------------------------------------------------
# Make panel widths identical across plots
# ------------------------------------------------
library(grid)

g_top <- ggplotGrob(p_top)
g_mid <- ggplotGrob(p_mid)
g_bot <- ggplotGrob(p_bot)

# Compute max widths across the grobs (all columns)
max_widths <- unit.pmax(g_top$widths, g_mid$widths, g_bot$widths)

# Apply the max widths back to each grob
g_top$widths <- max_widths
g_mid$widths <- max_widths
g_bot$widths <- max_widths

# Arrange with matched widths
grid.arrange(g_top, g_mid, g_bot, ncol = 1)

