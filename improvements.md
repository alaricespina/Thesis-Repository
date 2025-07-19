# Repository Optimization Recommendations

## Overview
This document provides comprehensive recommendations for optimizing the thesis repository by removing redundant files, archiving experimental code, and consolidating duplicate implementations. The current repository is **789MB** and can be reduced by **400-500MB (50-60%)** while preserving all production functionality.

## 🎯 Priority 1: Remove Duplicate Files (Immediate ~50MB savings)

### Identical Sensor Files
**Problem**: Same sensor code exists in multiple locations
- `/BMP180.py` (identical to `/Deployment - Deep Belief Networks/BMP180.py`)
- `/DHT11.py` (identical to `/Deployment - Deep Belief Networks/DHT11.py`) 
- `/HALL.py` (identical to `/Deployment - Deep Belief Networks/HALL.py`)

**Action**: 
- Keep only the versions in `/Deployment - Deep Belief Networks/`
- Remove root directory duplicates
- Update import paths if needed

### Duplicate Data Files (Major storage impact)
**Problem**: Yearly weather data (2001-2024) stored in 3 identical locations:
- `/Data Source Files/[YEAR].csv` 
- `/Deployment - Auto Regressive/Raw Atmospheric Data/[YEAR].csv`
- `/Deployment - Deep Belief Networks/Data/Yearly/[YEAR].csv`

**Action**:
- Keep only one centralized location: `/Data Source Files/`
- Remove the other 48 duplicate files
- Update scripts to reference single data source

### Duplicate Model Files
**Problem**: Same models stored multiple times
- `DBNFinalModel.keras` (2 locations)
- `NO_DBN_BI_RNN_50.keras` (2 locations)

**Action**:
- Keep only deployment versions
- Remove experimental copies

## 🎯 Priority 2: Archive Experimental Code (~218MB savings)

### Archive Entire Experimental Directories
**Move to archive branch or separate directory:**

```
archive/
├── auto-regressive-experiments/     (from Experimentation - Auto Regressive/)
├── deep-belief-network-experiments/ (from Experimentation - Deep Belief Networks/)
├── training-results/               (Training logs and old results)
└── deprecated-code/                (Deprecated GUI implementations)
```

### Specific Files to Archive:

#### Experimental Model Files (218MB)
- `Experimentation - Deep Belief Networks/Final Model Files/1 - KERAS MODELS/` (19MB, 36 files)
- `Experimentation - Deep Belief Networks/Final Model Files/2 - SCIKIT MODELS/` (181MB, 42 files)
- `Experimentation - Deep Belief Networks/Final Model Files/OLD TF TRAINING/` (18MB, 36 files)

#### Experimental Notebooks (25 files)
- All `.ipynb` files in `Experimentation - Auto Regressive/` (10 notebooks)
- All `.ipynb` files in `Experimentation - Deep Belief Networks/` (15 notebooks)

#### Training Results and Logs
- `Experimentation - Deep Belief Networks/Training Results/` (6 .txt files)
- Various `.pkl` accuracy files from experiments

## 🎯 Priority 3: Clean Up Development Files

### Remove Deprecated GUI Files
**Location**: `/Deployment - Deep Belief Networks/Deprecated/`
- `MainGUI.py`
- `PyQtGUISensorOnly.py` 
- `QtTest.py`
- `SimpleGUI.py`
- `Test.py`
- `VerySimpleGUI.py`

### Remove Temporary Files
- `/Deployment - Deep Belief Networks/tempCodeRunnerFile.py`
- Any other `tempCodeRunnerFile.py` instances

### Consider Removing Node Modules (~200MB)
- `/Simulator/node_modules/` can be regenerated with `npm install`

## 🎯 Priority 4: Consolidate Code Duplication

### GUI Implementation Issues Found
**Problem**: Two different `FinalGUI.py` versions:
- Root `/FinalGUI.py` - imports from `CalendarTest`
- `/Deployment - Deep Belief Networks/FinalGUI.py` - imports from `CalendarWidgetClass`

**Action**: 
- Determine which is the current production version
- Remove or rename the outdated version
- Standardize calendar widget imports

### Consolidate Utility Files
**Multiple implementations of similar functionality:**
- Calendar widgets: `Calendar.py`, `CalendarTest.py`, `CalendarWidgetClass.py`
- RBM implementations: Multiple versions in experimental folders

**Action**:
- Keep only the production-ready versions
- Archive experimental implementations

## 📁 Recommended Final Repository Structure

```
thesis-repository/
├── deployment/
│   ├── auto-regressive/          (Current Deployment - Auto Regressive)
│   ├── deep-belief-networks/     (Current Deployment - Deep Belief Networks)
│   └── shared/                   (Common sensor files, utilities)
├── data/                         (Centralized data storage)
├── server/                       (Current Server folder)
├── docs/                         (Documentation, if needed)
└── README.md                     (Project documentation)
```

## 🚀 Implementation Strategy

### Option 1: Git Branch Archive (Recommended)
```bash
# Create archive branch
git checkout -b archive-experimental
git add "Experimentation - Auto Regressive/"
git add "Experimentation - Deep Belief Networks/"
git commit -m "Archive experimental files and training results"
git push origin archive-experimental

# Return to main and clean up
git checkout main
git rm -r "Experimentation - Auto Regressive/"
git rm -r "Experimentation - Deep Belief Networks/"
git rm -r "Deployment - Deep Belief Networks/Deprecated/"
```

### Option 2: Archive Directory
Create an `archive/` folder in the repository and move experimental files there.

## 📊 Expected Results

### Storage Reduction
- **Experimental models**: ~218MB
- **Duplicate data files**: ~50MB  
- **Deprecated code**: ~30MB
- **Node modules** (optional): ~200MB
- **Total reduction**: 400-500MB (50-60% of current size)

### Repository Benefits
- ✅ Faster clone times
- ✅ Clearer project structure  
- ✅ Easier navigation for new developers
- ✅ Separated production from experimental code
- ✅ Preserved experimental work in archive for reference

### Production Code Preserved
- All deployment functionality maintained
- Sensor integration code kept
- Final trained models preserved
- GUI applications remain functional
- Server implementation intact

## ⚠️ Before Implementation

1. **Test current functionality** to ensure everything works
2. **Backup the repository** or create the archive branch first
3. **Update documentation** to reference new structure
4. **Verify import paths** after moving files
5. **Test deployment** after cleanup

## 🔍 Additional Cleanup Opportunities

### Code Quality Improvements
- Standardize import statements
- Remove unused imports
- Fix typos in comments and print statements
- Consolidate similar utility functions
- Add proper error handling

### Documentation
- Create proper README.md
- Document deployment procedures  
- Add model usage instructions
- Include hardware setup guide

This optimization will result in a cleaner, more maintainable repository while preserving all essential functionality and experimental work for future reference.