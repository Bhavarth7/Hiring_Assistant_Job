# 100B Hiring Assistant

A comprehensive, local-first hiring assistant application that helps recruiters and hiring managers analyze candidate data, score candidates, enforce diversity, and select top candidates for their teams.

## 🎯 Project Overview

The 100B Hiring Assistant is a sophisticated tool designed to streamline the hiring process by providing data-driven insights into candidate pools. It processes candidate submissions, applies intelligent scoring algorithms, and helps build diverse, high-performing teams.

## ✨ Key Features

### 🔍 **Smart Candidate Analysis**
- **Multi-dimensional scoring** based on skills, experience, education, and company background
- **Role-based categorization** (Engineering, Data, Product, Design, GTM, Operations)
- **Seniority detection** from job titles and experience
- **Impact assessment** based on achievements and metrics
- **Geographic diversity** enforcement and regional preference options

### 📊 **Data Visualization & Insights**
- Salary expectation analysis and distribution
- Skills frequency analysis
- Geographic distribution of candidates
- Education level breakdown
- Previous role analysis
- Interactive charts and visualizations

### 🎯 **Intelligent Candidate Selection**
- **Auto-select Top 5** candidates with diversity enforcement
- **Customizable filters** for score thresholds and role preferences
- **Region diversity preference** to ensure balanced team composition
- **Manual candidate review** with detailed profiles

### 🔒 **Privacy & Security**
- **100% local processing** - no data leaves your machine
- **No external API calls** or cloud dependencies
- **Secure file handling** for sensitive candidate information

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Clone or download** the project files to your local machine

2. **Install Python dependencies:**
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn flask
   ```

3. **Navigate to the project directory:**
   ```bash
   cd M_project/F2
   ```

### Running the Application

#### Option 1: Web Interface (Recommended)
```bash
python app.py
```
Then open your browser and navigate to `http://localhost:5000`

#### Option 2: Jupyter Notebook Analysis
```bash
cd F22
jupyter notebook 100B_Jobs.ipynb
```

## 📁 Project Structure

```
M_project/F2/
├── app.py                 # Main Flask application
├── index.html            # Web interface
├── styles.css            # Styling and UI components
├── F22/                  # Analysis and data processing
│   ├── 100B_Jobs.ipynb  # Jupyter notebook for data analysis
│   └── form-submissions.json  # Sample candidate data
└── README.md             # This file
```

## 🎮 Usage Guide

### 1. **Upload Candidate Data**
- Upload a JSON file containing candidate submissions
- Supports both JSON arrays and JSON Lines format
- The system will automatically process and score all candidates

### 2. **Review Candidate Scores**
- View comprehensive scoring breakdowns
- Analyze skills, experience, and background metrics
- Filter candidates by minimum score thresholds

### 3. **Apply Filters**
- Set minimum score requirements
- Select specific roles to include
- Enable/disable region diversity preferences

### 4. **Select Top Candidates**
- Use **Auto-select Top 5** for intelligent team building
- Manually review and adjust selections
- Export final picks to CSV format

### 5. **Deep Dive Analysis**
- Select individual candidates for detailed review
- Analyze specific aspects of candidate profiles
- Make informed hiring decisions

## 📊 Data Format

The application expects candidate data in the following JSON structure:

```json
[
  {
    "name": "Candidate Name",
    "location": "City, Country",
    "annual_salary_expectation": {
      "full-time": "$95,000"
    },
    "education": {
      "highest_level": "Bachelor's Degree"
    },
    "skills": ["Python", "React", "AWS"],
    "work_experiences": [
      {
        "roleName": "Senior Software Engineer",
        "company": "Tech Company"
      }
    ]
  }
]
```

## 🧠 Scoring Algorithm

The candidate scoring system evaluates multiple dimensions:

- **Skills Match** (40%): Alignment with role-specific skill requirements
- **Experience Level** (25%): Seniority and relevant work history
- **Company Background** (15%): Previous company reputation and scale
- **Education** (10%): Academic qualifications and institution quality
- **Geographic Diversity** (10%): Regional balance for team composition

## 🎨 Customization

### Adding New Roles
Modify the `ROLE_KEYWORDS` dictionary in `app.py` to include new role categories and their associated skills.

### Adjusting Scoring Weights
Modify the scoring algorithm in the `score_candidate` function to adjust the importance of different factors.

### Styling Changes
Edit `styles.css` to customize the visual appearance of the application.

## 🔧 Troubleshooting

### Common Issues

1. **Port already in use**: Change the port in `app.py` or stop other services using port 5000
2. **Missing dependencies**: Ensure all required Python packages are installed
3. **File upload errors**: Check that your JSON file is properly formatted

### Performance Tips

- For large datasets (>1000 candidates), consider processing in batches
- Use the notebook version for detailed analysis of smaller candidate pools
- Enable browser caching for better performance with repeated uploads

## 🤝 Contributing

This project is designed for local use and customization. Feel free to:

- Modify the scoring algorithms for your specific needs
- Add new visualization types
- Enhance the UI/UX components
- Extend the role categorization system

## 📄 License

This project is provided as-is for educational and business use. No warranty is provided.

## 🆘 Support

For questions or issues:

1. Check the troubleshooting section above
2. Review the code comments for implementation details
3. Examine the sample data format for guidance

---

**Built with ❤️ for modern hiring teams**

*Local-first, privacy-focused, and data-driven candidate selection.*
