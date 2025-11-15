# MLOps Course - Complete Solutions

This directory contains **complete, working solutions** for all 25 labs and modules in the MLOps Mastery course.

---

## 📚 Solutions Overview

### **Phase 1: Foundations** (3 labs)
- [`lab1_1_solution/`](phase1/lab1_1_solution/) - First Airflow DAG
- [`lab1_2_solution/`](phase1/lab1_2_solution/) - PyTorch Training Script
- [`lab1_3_solution/`](phase1/lab1_3_solution/) - Config-Driven Training

### **Phase 2: Data & Pipelines** (4 labs)
- [`lab2_1_solution/`](phase2/lab2_1_solution/) - ETL Pipeline
- [`lab2_2_solution/`](phase2/lab2_2_solution/) - Feature Engineering
- [`lab2_3_solution/`](phase2/lab2_3_solution/) - Data Quality Checks
- [`lab2_4_solution/`](phase2/lab2_4_solution/) - Scheduled Pipeline

### **Phase 3: Modeling & Training** (4 labs)
- [`lab3_1_solution/`](phase3/lab3_1_solution/) - Tabular Classifier
- [`lab3_2_solution/`](phase3/lab3_2_solution/) - Training DAG
- [`lab3_3_solution/`](phase3/lab3_3_solution/) - Experiment Tracking
- [`lab3_4_solution/`](phase3/lab3_4_solution/) - Two-Tower Model

### **Phase 4: Deployment & Monitoring** (5 labs)
- [`lab4_1_solution/`](phase4/lab4_1_solution/) - Model Serving (FastAPI)
- [`lab4_2_solution/`](phase4/lab4_2_solution/) - Batch Inference
- [`lab4_3_solution/`](phase4/lab4_3_solution/) - Monitoring & Drift
- [`lab4_4_solution/`](phase4/lab4_4_solution/) - Retraining Pipeline
- [`lab4_5_solution/`](phase4/lab4_5_solution/) - Complete System

### **Capstone: Mini Feed Ranking System** (7 modules)
- [`module1_solution/`](capstone/module1_solution/) - System Design
- [`module2_solution/`](capstone/module2_solution/) - Data Pipeline
- [`module3_solution/`](capstone/module3_solution/) - Training Pipeline
- [`module4_solution/`](capstone/module4_solution/) - Model Serving
- [`module5_solution/`](capstone/module5_solution/) - Monitoring
- [`module6_solution/`](capstone/module6_solution/) - Retraining
- [`module7_solution/`](capstone/module7_solution/) - Integration

---

## 🎯 How to Use Solutions

### **For Self-Study**

1. **Try the lab first**: Attempt the lab on your own
2. **Get stuck?**: Check the solution for hints
3. **Compare**: Review your solution against the provided one
4. **Learn**: Understand the patterns and best practices

### **Running Solutions**

Each solution directory contains:
- **Complete Python code** - Ready to run
- **README.md** - How to run, expected output
- **Configuration files** - YAML configs where needed
- **Test scripts** - To verify everything works

```bash
# General pattern to run any solution
cd solutions/phase1/lab1_1_solution
cat README.md  # Read instructions
python <script_name>.py  # Run the solution
```

### **Important Notes**

⚠️ **Don't skip labs**: Solutions are meant as reference, not shortcuts
✅ **Learn the patterns**: Understand *why* code is written this way
✅ **Adapt, don't copy**: Use solutions to learn, then write your own
✅ **Debug with solutions**: If stuck, compare your code to the solution

---

## 📁 Solution Structure

Each solution follows this pattern:

```
labX_Y_solution/
├── README.md              # How to run, what it does
├── <main_script>.py       # Primary implementation
├── config.yaml            # Configuration (if needed)
├── requirements.txt       # Extra dependencies (if any)
└── tests/                 # Test scripts (some labs)
```

---

## 🚀 Quick Start

### **Phase 1, Lab 1.1 Example**

```bash
# Navigate to Phase 1, Lab 1.1 solution
cd solutions/phase1/lab1_1_solution

# Read the README
cat README.md

# Copy DAG to your Airflow dags folder
cp hello_airflow.py ../../dags/

# Trigger in Airflow UI or CLI
airflow dags trigger hello_airflow
```

### **Phase 1, Lab 1.2 Example**

```bash
# Navigate to PyTorch training solution
cd solutions/phase1/lab1_2_solution

# Run training
python train_mnist.py

# Test the model
python test_mnist.py
```

---

## ✨ Solution Quality

All solutions feature:

✅ **Production-quality code** - Proper error handling, logging
✅ **Best practices** - Follows MLOps standards
✅ **Well-commented** - Explains key concepts
✅ **Complete** - Nothing left as "exercise for reader"
✅ **Tested** - Verified to work
✅ **Educational** - Written to teach, not just work

---

## 📖 Learning Path

### **Beginner Approach**
1. Read the lab instructions
2. Attempt on your own
3. Check solution if stuck
4. Compare and learn

### **Intermediate Approach**
1. Attempt the lab
2. Only look at solution structure (file organization)
3. Implement yourself
4. Compare implementations

### **Advanced Approach**
1. Complete the lab independently
2. Review solution for alternative approaches
3. Identify optimizations
4. Implement improvements

---

## 🔍 What's Included Per Phase

### **Phase 1 Solutions**
- Complete Airflow DAGs with tasks and dependencies
- Full PyTorch training with checkpointing and evaluation
- Config-driven training with experiment tracking
- **~500 lines of Python code**

### **Phase 2 Solutions**
- ETL pipeline with data generation and validation
- Feature engineering with train/val/test splits
- Data quality validation framework
- Scheduled pipelines with date partitioning
- **~800 lines of Python code**

### **Phase 3 Solutions**
- Tabular classification models
- Training orchestrated via Airflow
- Experiment tracking implementation
- Two-tower ranking models
- **~1000 lines of Python code**

### **Phase 4 Solutions**
- FastAPI model serving application
- Batch inference pipelines
- Drift detection and monitoring
- Automated retraining logic
- End-to-end system integration
- **~1200 lines of Python code**

### **Capstone Solutions**
- Complete feed ranking system
- Full ML lifecycle automation
- Production-grade components
- **~2000 lines of Python code**

**Total**: ~5,500 lines of production-quality Python code!

---

## 💡 Tips for Maximum Learning

### **Do's**
✅ Struggle first, then check solutions
✅ Type code manually (don't copy-paste)
✅ Modify solutions to experiment
✅ Read all comments carefully
✅ Run every solution to see it work

### **Don'ts**
❌ Copy-paste without understanding
❌ Skip directly to solutions
❌ Treat as boilerplate code
❌ Ignore the README files
❌ Rush through without experimenting

---

## 🛠 Troubleshooting

### **Solution doesn't run?**

1. Check you're in the right directory
2. Verify all dependencies installed: `pip install -r ../../setup/requirements.txt`
3. Check Airflow is running (for DAG solutions)
4. Read the solution's README for prerequisites

### **Different output than expected?**

- Solutions use random seeds where possible for reproducibility
- Some outputs (like timestamps) will naturally differ
- Model accuracy may vary slightly due to randomness

### **Want to modify a solution?**

1. Copy it to your working directory
2. Make modifications
3. Test thoroughly
4. Compare behavior to original

---

## 📊 Solution Statistics

| Phase | Labs | Python Files | Lines of Code | README Docs |
|-------|------|--------------|---------------|-------------|
| Phase 1 | 3 | 6 | ~500 | 3 |
| Phase 2 | 4 | 8 | ~800 | 4 |
| Phase 3 | 4 | 10 | ~1000 | 4 |
| Phase 4 | 5 | 12 | ~1200 | 5 |
| Capstone | 7 | 15 | ~2000 | 7 |
| **Total** | **23** | **51** | **~5500** | **23** |

---

## 🎓 Using Solutions for Interview Prep

Solutions can help you prepare for ML/MLOps interviews:

1. **Code Review Practice**: Review solutions as if in an interview
2. **System Design**: Study capstone architecture patterns
3. **Trade-offs**: Understand design decisions (documented in READMEs)
4. **Best Practices**: Learn production patterns
5. **Talking Points**: Use solutions as talking points for projects

---

## 📞 Need Help?

If solutions aren't working:

1. **Check the lab's README** - Specific troubleshooting there
2. **Verify environment** - Run the verification script from main README
3. **Check dependencies** - All labs need `setup/requirements.txt` installed
4. **Review docs** - Each phase has overview documentation

---

## 🎯 Success Criteria

You're using solutions effectively if you:

✅ Can explain every line of code in the solution
✅ Can modify solutions to add features
✅ Understand trade-offs made in the implementation
✅ Can apply patterns to new problems
✅ Successfully run all solutions

---

## 🚀 Next Steps

1. **Start with Phase 1, Lab 1.1**: [`phase1/lab1_1_solution/`](phase1/lab1_1_solution/)
2. **Read each lab's README** before running
3. **Experiment** with the code
4. **Build understanding** incrementally
5. **Complete the capstone** with confidence!

---

**Remember**: Solutions are learning tools, not shortcuts. The real value is in understanding *why* the code is written this way, not just *what* it does.

**Happy Learning!** 🎓
