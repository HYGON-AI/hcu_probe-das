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

### dump_enable动态控制示例

```json
{
    "task": "statistics",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",
    "dump_enable": false,
    "statistics": {
        "summary_mode": "statistics"
    }
}
```

> 说明：`dump_enable`仅在需要动态开关dump时配置。运行中可将`dump_enable`从`false`改为`true`（或反向修改）实现动态开关，json中其他字段修改也能生效。

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
