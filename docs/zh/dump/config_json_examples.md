# config.json配置样例

以下示例包含当前支持的所有场景可配置的完整参数。

## PyTorch场景

### task配置为statistics

```json
{
    "task": "statistics",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",
    "async_dump": false,

    "statistics": {
        "scope": [], 
        "list": [],
        "tensor_list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
```

### task配置为tensor

```json
{
    "task": "tensor",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",
    "async_dump": false,

    "tensor": {
        "scope": [],
        "list":[],
        "data_mode": ["all"],
        "bench_path": "/home/bench_data_dump",
        "summary_mode": "md5",
        "diff_nums": 5        
    }
}
```

### task配置为acc_check

```json
{
    "task": "acc_check",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",

    "acc_check": {
        "white_list": [],
        "black_list": [],
        "error_data_path": "./"
    }
}
```

### task配置为structure

```json
{
    "task": "structure",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "mix"
}
```

## MindSpore静态图场景

### task配置为statistics

```json
{
    "task": "statistics",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L2",

    "statistics": {
        "list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
```

### task配置为tensor

```json
{
    "task": "tensor",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L2",

    "tensor": {
        "list":[],
        "data_mode": ["all"]
    }
}
```

### task配置为overflow_check

```json
{
    "task": "overflow_check",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L2",

    "overflow_check": {
        "check_mode": "all"
    }
}
```

### task配置为exception_dump

```json
{
    "task": "exception_dump",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L2"
}
```

## MindSpore动态图场景

### task配置为statistics

```json
{
    "task": "statistics",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",

    "statistics": {
        "scope": [], 
        "list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
```

### task配置为tensor

```json
{
    "task": "tensor",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",

    "tensor": {
        "scope": [],
        "list":[],
        "data_mode": ["all"]
    }
}
```

### task配置为structure

```json
{
    "task": "structure",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "mix"
}
```

### task配置为exception_dump

```json
{
    "task": "exception_dump",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L2"
}
```
