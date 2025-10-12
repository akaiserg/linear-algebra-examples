# Data Science & Machine Learning Case Study
## Vector Addition in Recommendation Systems

### Overview
This case study demonstrates how vector addition is used in real-world Data Science and Machine Learning applications, specifically in recommendation systems.

### Problem Statement
Imagine you're building a movie recommendation system for Netflix. You need to combine multiple factors to predict how much a user will like a movie:

1. **User Preferences**: What genres the user typically enjoys
2. **Movie Features**: What genres the movie belongs to
3. **User Behavior**: How the user has rated similar movies
4. **Contextual Factors**: Time of day, device used, etc.

### The Vector Addition Approach

Instead of treating each factor separately, we can represent each factor as a vector and combine them using **true vector addition** (element-wise addition) to get a combined feature vector, which then gives us a final recommendation score.

**Key Point**: This uses vector addition `[a, b, c] + [d, e, f] = [a+d, b+e, c+f]`, not dot product!

### Files in this folder:
- `recommendation_system.py` - Python implementation of the recommendation system
- `README.md` - This documentation file

### Key Concepts Demonstrated:
- **Feature Vectors**: Representing user preferences and movie features as vectors
- **Vector Addition**: Element-wise addition of vectors `[a,b,c] + [d,e,f] = [a+d,b+e,c+f]`
- **Combined Feature Vectors**: Creating new vectors by adding user preferences + movie features
- **Weighted Vector Combinations**: Different importance levels for different factors
- **Real-world Application**: How vector addition is used in production ML systems

### Learning Objectives:
1. Understand how vectors represent features in ML
2. See how **vector addition** (not dot product) combines multiple factors
3. Learn the difference between vector addition and dot product
4. Learn about weighted vector combinations
5. Apply linear algebra concepts to real ML problems

### Next Steps:
1. Run the Python file to see the recommendation system in action
2. Modify the weights and features to see how recommendations change
3. Experiment with different user profiles and movie features
4. Understand how this scales to real recommendation systems
