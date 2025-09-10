import torch

print("=" * 50)
print(f"PyTorch 版本: {torch.__version__}")
print("=" * 50)

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA 可用: {'✅ 是' if cuda_available else '❌ 否'}")

if cuda_available:
    # 显示 CUDA 设备数量
    device_count = torch.cuda.device_count()
    print(f"\nCUDA 设备数量: {device_count}")
    
    # 显示每个设备的信息
    for i in range(device_count):
        print(f"\n设备 #{i}: {torch.cuda.get_device_name(i)}")
        print(f"  计算能力: {torch.cuda.get_device_capability(i)}")
        print(f"  内存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # 测试张量计算
    print("\n测试张量计算:")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn(10000, 10000, device=device)
    b = torch.randn(10000, 10000, device=device)
    c = torch.matmul(a, b)
    print(f"  矩阵乘法完成! 结果形状: {c.shape}")
    print(f"  首元素值: {c[0, 0].item():.4f}")
    
    # 基准测试
    print("\n基准测试 (CPU vs GPU):")
    import time
    
    # 在 CPU 上测试
    start_time = time.time()
    a_cpu = torch.randn(10000, 10000)
    b_cpu = torch.randn(10000, 10000)
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    print(f"  CPU 时间: {cpu_time:.4f} 秒")
    
    # 在 GPU 上测试
    if torch.cuda.is_available():
        start_time = time.time()
        a_gpu = a_cpu.to(device)
        b_gpu = b_cpu.to(device)
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()  # 等待所有 GPU 操作完成
        gpu_time = time.time() - start_time
        print(f"  GPU 时间: {gpu_time:.4f} 秒")
        print(f"  加速比: {cpu_time/gpu_time:.2f}x")
else:
    print("\n❌ CUDA 不可用，可能原因:")
    print("  - 未安装 CUDA 版本的 PyTorch")
    print("  - 系统没有 NVIDIA GPU")
    print("  - NVIDIA 驱动程序未安装或版本过旧")
    print("  - CUDA 工具包未安装或与 PyTorch 版本不兼容")

print("\n" + "=" * 50)