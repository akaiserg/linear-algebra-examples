"""
Customer Segmentation - Simplified Version
=========================================

Vector Transpose in Feature Engineering (No Charts, No Dot Products)
This example shows the core transpose concepts without complex operations.
"""

import numpy as np

def create_simple_customer_data():
    """Create simple customer behavior data."""
    print("=== Simple Customer Data ===")
    
    # Product categories
    categories = ['Electronics', 'Clothing', 'Books', 'Sports']
    
    # Customer data: [Electronics, Clothing, Books, Sports]
    # Values represent engagement level (0-1 scale)
    customers = np.array([
        [0.9, 0.2, 0.1, 0.3],  # Customer 1: Tech enthusiast
        [0.1, 0.8, 0.6, 0.2],  # Customer 2: Fashion lover
        [0.2, 0.3, 0.9, 0.1],  # Customer 3: Bookworm
        [0.3, 0.4, 0.2, 0.9],  # Customer 4: Sports enthusiast
        [0.7, 0.6, 0.4, 0.5]   # Customer 5: Mixed interests
    ])
    
    print(f"Categories: {categories}")
    print(f"Customer data shape: {customers.shape}")
    print(f"Customer data:")
    for i, customer in enumerate(customers):
        print(f"Customer {i+1}: {customer}")
    
    return customers, categories

def basic_transpose_demo():
    """Demonstrate basic transpose operations."""
    print("\n=== Basic Transpose Demo ===")
    
    customers, categories = create_simple_customer_data()
    
    # Original data: customers as rows
    print(f"Original data shape: {customers.shape}")
    print("Perspective: Each row = one customer")
    print(f"Original data:\n{customers}")
    
    # Transpose: categories as rows
    categories_data = customers.T
    print(f"\nTransposed data shape: {categories_data.shape}")
    print("Perspective: Each row = one product category")
    print(f"Transposed data:\n{categories_data}")
    
    return customers, categories, categories_data

def category_analysis_simple():
    """Simple category analysis using transpose."""
    print("\n=== Simple Category Analysis ===")
    
    customers, categories, categories_data = basic_transpose_demo()
    
    print("Category Statistics:")
    print("-" * 40)
    
    for i, category in enumerate(categories):
        category_values = categories_data[i]
        mean_val = np.mean(category_values)
        max_val = np.max(category_values)
        min_val = np.min(category_values)
        
        print(f"{category:12} | Mean: {mean_val:.3f} | Max: {max_val:.3f} | Min: {min_val:.3f}")
    
    # Find most popular category
    category_means = np.mean(categories_data, axis=1)
    most_popular_idx = np.argmax(category_means)
    
    print(f"\nMost Popular Category: {categories[most_popular_idx]} "
          f"(average: {category_means[most_popular_idx]:.3f})")

def customer_segmentation_simple():
    """Simple customer segmentation."""
    print("\n=== Simple Customer Segmentation ===")
    
    customers, categories, categories_data = basic_transpose_demo()
    
    # Method 1: Find each customer's favorite category
    print("Customer Segments (by favorite category):")
    print("-" * 45)
    
    for i, customer in enumerate(customers):
        favorite_idx = np.argmax(customer)
        favorite_category = categories[favorite_idx]
        favorite_score = customer[favorite_idx]
        
        print(f"Customer {i+1}: {favorite_category:12} (score: {favorite_score:.3f})")
    
    # Method 2: Group by engagement level
    print(f"\nEngagement Level Groups:")
    print("-" * 30)
    
    total_engagement = np.sum(customers, axis=1)
    
    for i, total in enumerate(total_engagement):
        if total > 2.0:
            level = "High"
        elif total > 1.5:
            level = "Medium"
        else:
            level = "Low"
        
        print(f"Customer {i+1}: {level:6} engagement (total: {total:.3f})")

def feature_engineering_simple():
    """Simple feature engineering using transpose."""
    print("\n=== Simple Feature Engineering ===")
    
    customers, categories, categories_data = basic_transpose_demo()
    
    # Feature 1: Category popularity (using transpose)
    print("Feature 1: Category Popularity")
    category_popularity = np.mean(categories_data, axis=1)
    
    for i, category in enumerate(categories):
        print(f"{category}: {category_popularity[i]:.3f}")
    
    # Feature 2: Customer diversity (how spread out their preferences are)
    print(f"\nFeature 2: Customer Preference Diversity")
    customer_diversity = np.std(customers, axis=1)
    
    for i, diversity in enumerate(customer_diversity):
        if diversity > 0.3:
            diversity_level = "Diverse"
        elif diversity > 0.2:
            diversity_level = "Moderate"
        else:
            diversity_level = "Focused"
        
        print(f"Customer {i+1}: {diversity_level:8} (std: {diversity:.3f})")
    
    # Feature 3: Enhanced customer profiles
    print(f"\nFeature 3: Enhanced Customer Profiles")
    print("Original + Total Engagement + Diversity")
    
    enhanced_profiles = np.column_stack([
        customers,  # Original preferences
        np.sum(customers, axis=1, keepdims=True),  # Total engagement
        np.std(customers, axis=1, keepdims=True)   # Diversity
    ])
    
    print(f"Enhanced profiles shape: {enhanced_profiles.shape}")
    print("Features: [Electronics, Clothing, Books, Sports, Total, Diversity]")
    
    for i, profile in enumerate(enhanced_profiles):
        print(f"Customer {i+1}: {profile}")

def business_insights_simple():
    """Simple business insights."""
    print("\n=== Simple Business Insights ===")
    
    customers, categories, categories_data = basic_transpose_demo()
    
    # Insight 1: Category ranking
    category_means = np.mean(categories_data, axis=1)
    category_ranking = np.argsort(category_means)[::-1]
    
    print("Category Performance Ranking:")
    print("-" * 35)
    for i, idx in enumerate(category_ranking):
        print(f"{i+1}. {categories[idx]:12} (avg: {category_means[idx]:.3f})")
    
    # Insight 2: Customer value
    customer_totals = np.sum(customers, axis=1)
    high_value_customers = np.argsort(customer_totals)[-2:]
    
    print(f"\nTop 2 High-Value Customers:")
    print("-" * 30)
    for i, customer_idx in enumerate(high_value_customers):
        print(f"{i+1}. Customer {customer_idx+1} (total: {customer_totals[customer_idx]:.3f})")
    
    # Insight 3: Category combinations
    print(f"\nCategory Combination Analysis:")
    print("-" * 35)
    
    # Find customers who like multiple categories
    multi_category_customers = np.sum(customers > 0.5, axis=1)
    
    for i, count in enumerate(multi_category_customers):
        if count >= 2:
            print(f"Customer {i+1}: Likes {count} categories (multi-category shopper)")

def transpose_benefits():
    """Show the benefits of using transpose."""
    print("\n=== Benefits of Transpose ===")
    
    customers, categories, categories_data = basic_transpose_demo()
    
    print("Why transpose is useful:")
    print("1. Switch perspective: Customer view ↔ Category view")
    print("2. Calculate category statistics easily")
    print("3. Compare customers across categories")
    print("4. Find patterns in data")
    
    print(f"\nExample: Category analysis without transpose (harder):")
    # Without transpose - need to extract each category manually
    electronics_scores = customers[:, 0]  # First column
    clothing_scores = customers[:, 1]     # Second column
    print(f"Electronics scores: {electronics_scores}")
    print(f"Clothing scores: {clothing_scores}")
    
    print(f"\nExample: Category analysis with transpose (easier):")
    # With transpose - each row is a category
    print(f"Electronics row: {categories_data[0]}")
    print(f"Clothing row: {categories_data[1]}")
    
    print(f"\nTranspose makes category analysis much simpler!")

def real_world_example():
    """Simple real-world example."""
    print("\n=== Real-World Example ===")
    
    print("Imagine you're analyzing Netflix user data:")
    print("- Each row = one user's preferences")
    print("- Each column = one movie genre")
    print("- Transpose lets you analyze genres across all users")
    
    # Simulate Netflix data
    genres = ['Action', 'Comedy', 'Drama', 'Horror']
    users = np.array([
        [0.9, 0.3, 0.2, 0.1],  # Action fan
        [0.2, 0.8, 0.6, 0.1],  # Comedy lover
        [0.1, 0.2, 0.9, 0.0],  # Drama enthusiast
        [0.3, 0.4, 0.3, 0.8]   # Horror fan
    ])
    
    print(f"\nUser preferences:")
    for i, user in enumerate(users):
        print(f"User {i+1}: {user}")
    
    # Transpose for genre analysis
    genre_data = users.T
    print(f"\nGenre analysis (transposed):")
    for i, genre in enumerate(genres):
        print(f"{genre}: {genre_data[i]}")
    
    # Find most popular genre
    genre_popularity = np.mean(genre_data, axis=1)
    most_popular = genres[np.argmax(genre_popularity)]
    print(f"\nMost popular genre: {most_popular}")

if __name__ == "__main__":
    print("Customer Segmentation - Simplified Version")
    print("Vector Transpose in Feature Engineering")
    print("=" * 50)
    
    # Run simplified examples
    basic_transpose_demo()
    category_analysis_simple()
    customer_segmentation_simple()
    feature_engineering_simple()
    business_insights_simple()
    transpose_benefits()
    real_world_example()
    
    print("\n" + "=" * 50)
    print("Simplified Case Study Complete!")
    print("Key Takeaway: Transpose helps analyze data from different perspectives!")
