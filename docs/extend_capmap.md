Before retraining the classifier for a new cap type, you must first extend the cap mapping. This mapping associates cap's name with an integer label. The integer label used in model training.

### Current Cap Mapping

The current default cap mapping that can be found by runing `capfinder capmap list` command, and is as follows:

```
-99: cap_unknown
0: cap_0
1: cap_1
2: cap_2
3: cap_2-1
```

We strongly recommend not changing the mapping of existing caps. Instead, extend the existing mapping to accomodate new cap types as needed.

### Adding a New Cap Type


To add a new cap type, use the `capmap add` command:

```bash
capfinder capmap add <integer> <cap_name>
```

### Adding a New Cap Type

To add a new cap type, use the `capmap add` command:

```bash
capfinder capmap add 4 cap_0m6A
```

###  Verifying the New Cap Type
After adding the new cap type, you can verify it has been added correctly:

```
capfinder capmap list
```

You should see your new cap type in the list of current cap mappings:


```
Current cap mappings:
-99: cap_unknown
0: cap_0
1: cap_1
2: cap_2
3: cap_2-1
4: cap_0m6A
```
