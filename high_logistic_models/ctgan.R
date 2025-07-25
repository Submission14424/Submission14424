.libPaths("/mnt/ufs18/home-249/wangzun/anaconda3/envs/myr/lib/R/library")

library(glmnet)

# 文件路径配置
file_paths <- list(
  ctgan_file_path = "ctgan.csv",
  file_path_test = "test.csv"
)

# 读取数据函数（适用于包含特征和标签的文件）
read_target_data <- function(file_path) {
  data <- read.csv(file_path, header = FALSE)
  x <- as.matrix(data[, 1:511])
  y <- as.numeric(data[[512]])
  return(list(x = x, y = y))
}

# 读取测试数据
test_data <- read_target_data(file_paths$file_path_test)

# 读取训练数据 ctgan
ctgan_data <- read.csv(file_paths$ctgan_file_path, header = FALSE)

# 获取当前任务的索引
task_id <- as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))

# 定义切分比例
# 生成从 0.01 到 0.2，步长为 0.01 的序列
seq1 <- seq(from = 0.01, to = 0.3, by = 0.01)

# 生成从 0.3 到 1，步长为 0.1 的序列
seq2 <- seq(from = 0.4, to = 1, by = 0.1)
# 合并两个序列
split_ratios <- c(seq1, seq2)
# 选择当前任务的切分比例
ratio <- split_ratios[task_id]

# 批次大小
batch_size <- 200

# 存储结果
pre_errors <- c()

# 进行 100 次随机抽样并计算 prediction error
for (iter in 1:100) {
  seed <- task_id + iter
  set.seed(seed)
  # 随机抽样数据
  ctgan_sub_data <- ctgan_data[sample(nrow(ctgan_data), floor(nrow(ctgan_data) * ratio)), ]
  
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
  
  batches <- create_batches(ctgan_sub_data, batch_size)
  
  # 合并所有批次作为训练集
  train_x <- do.call(rbind, lapply(batches, function(batch) batch$x))
  train_y <- unlist(lapply(batches, function(batch) batch$y))
  
  # 使用 cv.glmnet 进行二分类模型训练，采用 type.measure = "class"
  fit.lasso <- cv.glmnet(x = train_x, y = train_y, family = "binomial", type.measure = "class")
  
  # 在测试集上预测（type = "class" 返回的是因子型）
  y.pred.lasso <- predict(fit.lasso, newx = test_data$x, type = "class")
  
  # 计算 prediction error
  pre_error_iter <- mean((as.numeric(y.pred.lasso) - test_data$y)^2)
  pre_errors <- c(pre_errors, pre_error_iter)
}

if (length(pre_errors) > 0) {
  avg_pre_error <- mean(pre_errors)
  
  # 保存为 CSV 文件
  file_name <- paste0("ctgan_", task_id, ".csv")
  write.csv(data.frame(avg_pre_error = avg_pre_error), file_name, row.names = FALSE)
}

cat("任务", task_id, "已完成并保存。\n")
