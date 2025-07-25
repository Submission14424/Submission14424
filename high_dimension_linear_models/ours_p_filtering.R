.libPaths("/mnt/ufs18/home-249/wangzun/anaconda3/envs/myr/lib/R/library")

# 加载所需包
library(data.table)
library(glmtrans)
library(glmnet)
library(dplyr)
library(purrr)
library(pbapply)
library(hdtrd)

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
  target_X <- as.matrix(target_data[, 1:511])
  target_Y <- as.numeric(target_data[[512]])
  return(list(X = target_X, Y = target_Y))
}

# 读取源数据函数
read_source_data <- function(file_path) {
  return(read.csv(file_path, header = FALSE))
}

# 读取数据
v1_target <- read_target_data(file_paths$v1_file_path_target)
v2_target <- read_target_data(file_paths$v2_file_path_target)
test_data <- read_target_data(file_paths$file_path_test)

# 合并已知数据
v1_v2_X_known <- rbind(v1_target$X, v2_target$X)
v1_v2_Y_known <- c(v1_target$Y, v2_target$Y)

# 读取源数据
v1_source_data <- read_source_data(file_paths$v1_file_path_source)
v2_source_data <- read_source_data(file_paths$v2_file_path_source)

# 初始化参数，实际数据并不需要
beta_0 <- c(0, 2, -1, 0.5, rep(0, 512 - 4))
test_1X <- cbind(1, test_data$X)


n_repeats <- 1000
errors <- numeric(n_repeats)

for (i in 1:n_repeats) {
  set.seed(i + 1)  # 保证每次种子不同
  
  # 训练 Lasso 模型（高维特征）
  fit.lasso <- cv.glmnet(x = v1_v2_X_known, y = v1_v2_Y_known)
  
  # 预测
  y.pred.lasso <- predict(fit.lasso, newx = test_data$X)
  
  # 计算 MSE
  pre_error_before <- mean((y.pred.lasso - test_data$Y)^2)
  errors[i] <- pre_error_before
}

# 输出结果
mean_pre_error_before <- mean(errors)
sd_pre_error_before <- sd(errors)

cat("Mean pre_error_before:", mean_pre_error_before, "\n")
cat("SD pre_error_before:", sd_pre_error_before, "\n")

# 获取当前任务的索引
task_id <- as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))

# 定义切分比例
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
estimate_errors_v1 <- c()
pre_errors_v2 <- c()
estimate_errors_v2 <- c()
betas_v1 <- list()
betas_v2 <- list()
pre_errors_v1_v2_combined <- c()
v1_v2_combined_estimate_error <- c()
len_v1 <- c()
len_v2 <- c()

# 批次划分函数
create_batches <- function(data, batch_size) {
  lapply(seq(1, nrow(data), by = batch_size), function(j) {
    end_row <- min(j + batch_size - 1, nrow(data))
    if (end_row < j) return(NULL)
    
    source_X <- as.matrix(data[j:end_row, 1:511, drop = FALSE])
    source_Y <- as.numeric(unlist(data[j:end_row, 512]))
    dimnames(source_X) <- NULL
    attributes(source_Y) <- NULL
    
    list(X = source_X, Y = source_Y)
  })
}

# 进行100次随机抽样并计算
for (iter in 1:100) {
  seed <- task_id + iter
  set.seed(seed)
  
  # 随机抽样数据
  v1_sub_data <- v1_source_data[sample(nrow(v1_source_data), floor(nrow(v1_source_data) * ratio)), ]
  v2_sub_data <- v2_source_data[sample(nrow(v2_source_data), floor(nrow(v2_source_data) * ratio)), ]
  
  # 为v1进行batch划分
  source_list_v1 <- create_batches(v1_sub_data, batch_size)
  D_training_v1 <- list(
    target = list(
      X = { X <- as.matrix(v2_target$X); dimnames(X) <- NULL; X },
      Y = { Y <- as.numeric(v2_target$Y); attributes(Y) <- NULL; Y }
    ),
    source = source_list_v1
  )
  
  # pvaltrans 检测
  v1_pval <- pvaltrans(
    target = D_training_v1$target, 
    source = D_training_v1$source, 
    delta0 = 0, 
    nsource = length(D_training_v1$source)
  )
  v1_u_sources <- which(v1_pval < 0.01)
  
  if (length(v1_u_sources) == 0) next
  
  # 筛选可转移的 sources
  v1_valid_sources <- D_training_v1$source[v1_u_sources]
  
  # 修改 D_training_v1$target 和 v1_valid_sources 的命名
  D_training_v1$target <- list(x = D_training_v1$target$X, y = D_training_v1$target$Y)
  v1_valid_sources <- lapply(v1_valid_sources, function(source) {
    list(x = source$X, y = source$Y)
  })
  
  # 使用glmtrans拟合
  fit_gaussian_v1 <- glmtrans(D_training_v1$target, v1_valid_sources, cores = 1,transfer.source.id="all")
  
  # 获取可转移的 sources
  v1_transferable_sources <- fit_gaussian_v1$transfer.source.id
  if (length(v1_transferable_sources) == 0) next
  
  # 为v2进行batch划分
  source_list_v2 <- create_batches(v2_sub_data, batch_size)
  D_training_v2 <- list(
    target = list(
      X = { X <- as.matrix(v1_target$X); dimnames(X) <- NULL; X },
      Y = { Y <- as.numeric(v1_target$Y); attributes(Y) <- NULL; Y }
    ),
    source = source_list_v2
  )
  
  # pvaltrans 检测
  v2_pval <- pvaltrans(
    target = D_training_v2$target, 
    source = D_training_v2$source, 
    delta0 = 0, 
    nsource = length(D_training_v2$source)
  )
  v2_u_sources <- which(v2_pval < 0.01)
  
  if (length(v2_u_sources) == 0) next
  
  # 筛选可转移的 sources
  v2_valid_sources <- D_training_v2$source[v2_u_sources]
  
  # 修改 D_training_v2$target 和 v2_valid_sources 的命名
  D_training_v2$target <- list(x = D_training_v2$target$X, y = D_training_v2$target$Y)
  v2_valid_sources <- lapply(v2_valid_sources, function(source) {
    list(x = source$X, y = source$Y)
  })
  
  # 使用glmtrans拟合
  fit_gaussian_v2 <- glmtrans(D_training_v2$target, v2_valid_sources, cores = 1,transfer.source.id="all")
  
  # 获取可转移的 sources
  v2_transferable_sources <- fit_gaussian_v2$transfer.source.id
  if (length(v2_transferable_sources) == 0) next
  
  # 计算预测误差
  pre_error_v1_generate_glmtrans <- mean((predict(fit_gaussian_v1, test_data$X) - test_data$Y)^2)
  v1_estimate_error <- mean((fit_gaussian_v1$beta - beta_0)^2)
  pre_errors_v1 <- c(pre_errors_v1, pre_error_v1_generate_glmtrans)
  estimate_errors_v1 <- c(estimate_errors_v1, v1_estimate_error)
  betas_v1 <- c(betas_v1, list(fit_gaussian_v1$beta))
  len_v1 <- c(len_v1, length(v1_transferable_sources))
  
  pre_error_v2_generate_glmtrans <- mean((predict(fit_gaussian_v2, test_data$X) - test_data$Y)^2)
  v2_estimate_error <- mean((fit_gaussian_v2$beta - beta_0)^2)
  pre_errors_v2 <- c(pre_errors_v2, pre_error_v2_generate_glmtrans)
  estimate_errors_v2 <- c(estimate_errors_v2, v2_estimate_error)
  betas_v2 <- c(betas_v2, list(fit_gaussian_v2$beta))
  len_v2 <- c(len_v2, length(v2_transferable_sources))
  
  # 计算组合误差
  combined_beta <- (fit_gaussian_v1$beta + fit_gaussian_v2$beta) / 2
  pre_error_v1_v2_combined_iter <- mean((test_1X %*% combined_beta  - test_data$Y)^2)
  pre_errors_v1_v2_combined <- c(pre_errors_v1_v2_combined, pre_error_v1_v2_combined_iter)
  v1_v2_combined_estimate_error_iter <- mean((combined_beta - beta_0)^2)
  v1_v2_combined_estimate_error <- c(v1_v2_combined_estimate_error, v1_v2_combined_estimate_error_iter)
}

# 计算每次循环后的平均误差
if (length(pre_errors_v1) > 0 && length(pre_errors_v2) > 0) {
  avg_pre_error_v1 <- mean(pre_errors_v1)
  avg_estimate_error_v1 <- mean(estimate_errors_v1)
  avg_pre_error_v2 <- mean(pre_errors_v2)
  avg_estimate_error_v2 <- mean(estimate_errors_v2)
  avg_len_v1 <- mean(len_v1)
  avg_len_v2 <- mean(len_v2)
  avg_len_v1v2 <- (avg_len_v1 + avg_len_v2)*batch_size
  
  avg_beta_v1 <- colMeans(do.call(rbind, betas_v1))
  avg_beta_v2 <- colMeans(do.call(rbind, betas_v2))
  
  avg_pre_error_v1_v2_combined <- mean(pre_errors_v1_v2_combined)
  avg_v1_v2_combined_estimate_error <- mean(v1_v2_combined_estimate_error)
  
  # 存储结果
  final_results <- list(
    avg_pre_error_v1 = avg_pre_error_v1,
    avg_estimate_error_v1 = avg_estimate_error_v1,
    avg_pre_error_v2 = avg_pre_error_v2,
    avg_estimate_error_v2 = avg_estimate_error_v2,
    avg_beta_v1 = avg_beta_v1,
    avg_beta_v2 = avg_beta_v2,
    avg_pre_error_v1_v2_combined = avg_pre_error_v1_v2_combined,
    avg_v1_v2_combined_estimate_error = avg_v1_v2_combined_estimate_error,
    avg_len_v1v2 = avg_len_v1v2,
    pre_error_before = 1.165964
  )
  
  # 保存为 CSV 文件
  file_name <- paste0("ugt_", task_id, ".csv")
  write.csv(final_results, file_name, row.names = FALSE)
}

# 打印完成
cat("任务", task_id, "已完成并保存。\n")
