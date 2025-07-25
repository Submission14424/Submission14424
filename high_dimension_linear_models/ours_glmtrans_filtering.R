.libPaths("/mnt/ufs18/home-249/wangzun/anaconda3/envs/myr/lib/R/library")

library(data.table)
library(glmtrans)
library(glmnet)
library(dplyr)
library(purrr)
# library(readr)
library(pbapply)



# 文件路径配置
file_paths <- list(
  v1_file_path_source = "v1_merged_result.csv",
  v2_file_path_source = "v2_merged_result.csv",
  v1_file_path_target = "v1.csv",
  v2_file_path_target = "v2.csv",
  file_path_test = "test.csv"
)

# 读取目标数据函数
read_target_data <- function(file_path) {
  target_data <- read.csv(file_path, header = FALSE)
  target_x <- as.matrix(target_data[, 1:511])
  target_y <- as.numeric(target_data[[512]])
  return(list(x = target_x, y = target_y))
}

# 读取目标数据
v1_target <- read_target_data(file_paths$v1_file_path_target)
v2_target <- read_target_data(file_paths$v2_file_path_target)
test_data <- read_target_data(file_paths$file_path_test)

# 合并已知数据
v1_v2_x_known <- rbind(v1_target$x, v2_target$x)
v1_v2_y_known <- c(v1_target$y, v2_target$y)

# 读取源数据函数
read_source_data <- function(file_path) {
  return(read.csv(file_path, header = FALSE))
}

# 读取源数据
v1_source_data <- read_source_data(file_paths$v1_file_path_source)
v2_source_data <- read_source_data(file_paths$v2_file_path_source)

# Lasso回归
fit_lasso <- cv.glmnet(x = v1_v2_x_known, y = v1_v2_y_known)
y_pred_lasso <- predict(fit_lasso, test_data$x)
beta_before <- coef(fit_lasso)
pre_error_before <- mean((y_pred_lasso - test_data$y)^2)

# 初始化参数，boston并不需要
# beta_0 <- c(0, 2, -1, 0.5)
test_1x <- cbind(1, test_data$x)

# 获取当前任务的索引
task_id <- as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))

# 定义切分比例
# 生成从 0.01 到 0.2，步长为 0.01 的序列
seq1 <- seq(from = 0.01, to = 0.3, by = 0.01)
seq2 <- seq(from = 0.32, to = 0.5, by = 0.02)
split_ratios <- c(seq1, seq2)
ratio <- split_ratios[task_id]

# 批次大小
batch_size <- 100

# 存储结果列表
final_results_list <- list()

# 初始化结果
results_v1 <- list()
results_v2 <- list()
pre_errors_v1 <- c()
# estimate_errors_v1 <- c()
pre_errors_v2 <- c()
# estimate_errors_v2 <- c()
# betas_v1 <- list()
# betas_v2 <- list()
pre_errors_v1_v2_combined <- c()
# v1_v2_combined_estimate_error <- c()
len_v1 <- c()
len_v2 <- c()

# 进行100次随机抽样并计算
for (iter in 1:100) {
  seed <- task_id + iter
  # 随机抽样数据
  set.seed(seed)
  v1_sub_data <- v1_source_data[sample(nrow(v1_source_data), floor(nrow(v1_source_data) * ratio)), ]
  v2_sub_data <- v2_source_data[sample(nrow(v2_source_data), floor(nrow(v2_source_data) * ratio)), ]
  
  # 批次划分函数
  create_batches <- function(data, batch_size) {
    lapply(seq(1, nrow(data), by = batch_size), function(j) {
      end_row <- min(j + batch_size - 1, nrow(data))
      if (end_row < j) return(NULL)
      
      source_x <- as.matrix(data[j:end_row, 1:511, drop = FALSE])
      source_y <- as.numeric(unlist(data[j:end_row, 512]))
      
      list(x = source_x, y = source_y)
    })
  }
  
  # 为v1进行batch划分
  source_list_v1 <- create_batches(v1_sub_data, batch_size)
  D_training_v1 <- list(
    target = list(x = v2_target$x, y = v2_target$y),
    source = source_list_v1
  )
  
  # 适配你的 glmtrans 函数（调整 cores 参数）
  fit_gaussian_v1 <- glmtrans(D_training_v1$target, D_training_v1$source, cores = 1)
  v1_transferable_sources <- fit_gaussian_v1$transfer.source.id
  
  if (length(v1_transferable_sources) == 0) next
  
  # 为v2进行batch划分
  source_list_v2 <- create_batches(v2_sub_data, batch_size)
  D_training_v2 <- list(
    target = list(x = v1_target$x, y = v1_target$y),
    source = source_list_v2
  )
  
  fit_gaussian_v2 <- glmtrans(D_training_v2$target, D_training_v2$source, cores = 1)
  v2_transferable_sources <- fit_gaussian_v2$transfer.source.id
  
  if (length(v2_transferable_sources) == 0) next
  
  
  pre_error_v1_generate_glmtrans <- mean((predict(fit_gaussian_v1, test_data$x) - test_data$y)^2)
  # v1_estimate_error <- mean(fit_gaussian_v1$beta - beta_0)^2
  pre_errors_v1 <- c(pre_errors_v1, pre_error_v1_generate_glmtrans)
  # estimate_errors_v1 <- c(estimate_errors_v1, v1_estimate_error)
  # betas_v1 <- c(betas_v1, list(fit_gaussian_v1$beta))
  len_v1 <- c(len_v1, length(v1_transferable_sources))
  
  pre_error_v2_generate_glmtrans <- mean((predict(fit_gaussian_v2, test_data$x) - test_data$y)^2)
  # v2_estimate_error <- mean(fit_gaussian_v2$beta - beta_0)^2
  pre_errors_v2 <- c(pre_errors_v2, pre_error_v2_generate_glmtrans)
  # estimate_errors_v2 <- c(estimate_errors_v2, v2_estimate_error)
  # betas_v2 <- c(betas_v2, list(fit_gaussian_v2$beta))
  len_v2 <- c(len_v2, length(v2_transferable_sources))
  
  # 计算组合误差
  pre_error_v1_v2_combined_iter <- mean((test_1x %*% ((fit_gaussian_v1$beta + fit_gaussian_v2$beta) / 2) - test_data$y)^2)
  # v1_v2_combined_estimate_error_iter <- mean(((fit_gaussian_v1$beta + fit_gaussian_v2$beta) / 2) - beta_0)^2
  
  pre_errors_v1_v2_combined <- c(pre_errors_v1_v2_combined, pre_error_v1_v2_combined_iter)
  # v1_v2_combined_estimate_error <- c(v1_v2_combined_estimate_error, v1_v2_combined_estimate_error_iter)
}

if (length(pre_errors_v1) > 0 && length(pre_errors_v2) > 0) {
  # 计算每次循环后的平均误差
  avg_pre_error_v1 <- mean(pre_errors_v1)
  # avg_estimate_error_v1 <- mean(estimate_errors_v1)
  avg_pre_error_v2 <- mean(pre_errors_v2)
  # avg_estimate_error_v2 <- mean(estimate_errors_v2)
  avg_len_v1 <- mean(len_v1)
  avg_len_v2 <- mean(len_v2)
  avg_len_v1v2 <- (avg_len_v1 + avg_len_v2)*batch_size
  
  # avg_beta_v1 <- colMeans(do.call(rbind, betas_v1))
  # avg_beta_v2 <- colMeans(do.call(rbind, betas_v2))
  
  avg_pre_error_v1_v2_combined <- mean(pre_errors_v1_v2_combined)
  # avg_v1_v2_combined_estimate_error <- mean(v1_v2_combined_estimate_error)
  
  # 存储结果
  final_results <- list(
    avg_pre_error_v1 = avg_pre_error_v1,
    # avg_estimate_error_v1 = avg_estimate_error_v1,
    avg_pre_error_v2 = avg_pre_error_v2,
    # avg_estimate_error_v2 = avg_estimate_error_v2,
    # avg_beta_v1 = avg_beta_v1,
    # avg_beta_v2 = avg_beta_v2,
    avg_pre_error_v1_v2_combined = avg_pre_error_v1_v2_combined,
    # avg_v1_v2_combined_estimate_error = avg_v1_v2_combined_estimate_error,
    avg_len_v1v2 = avg_len_v1v2,
    pre_error_before = pre_error_before
  )
  
  # 保存为 CSV 文件
  file_name <- paste0("glm_run_", task_id, ".csv")
  write.csv(final_results, file_name, row.names = FALSE)
}

# 打印完成
cat("任务", task_id, "已完成并保存。\n")
