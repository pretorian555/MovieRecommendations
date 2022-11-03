
"""
Created on Thu Oct 20 15:41:56 2022

@author: Milan Jendrisek

this script calcuates most similar movies based on the 25 million movie ratings dataset of 62k movies. was produced by GroupLens - 
a research lab in the Department of Computer Science and Engineering at the University of Minnesota. 
The script will output a list of most similar movies for each movie in scv format to be used with movie recommender website 

"""

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as func
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType
import sys


def load_dfs(file_name, file_type, infer_schema,first_row_is_header, delimiter):
    
    
    movieNamesSchema = StructType([ \
                               StructField("movieId", IntegerType(), True), \
                               StructField("movieTitle", StringType(), True) \
                               ])
    
    moviesSchema = StructType([ \
                     StructField("userId", IntegerType(), True), \
                     StructField("movieId", IntegerType(), True), \
                     StructField("rating", IntegerType(), True), \
                     StructField("timestamp", LongType(), True)])

# The applied options are for CSV files. For other file types, these will be ignored.

    if "movies" in file_name:
        schema = movieNamesSchema
    else:
        schema = moviesSchema

    if infer_schema:
        df = spark.read.format(file_type) \
            .option("inferSchema", infer_schema) \
            .option("header", first_row_is_header) \
            .option("sep", delimiter) \
            .load(file_name)
    else:
        df = spark.read.format(file_type) \
            .schema(schema) \
            .option("header", first_row_is_header) \
            .option("sep", delimiter) \
            .load(file_name)

    return df

##function that will create a dataframe of movie pairs, each row of the dataframe is a join of two movieIDs rated by the same user
##the result is a dataframe of movie pairs joined with each other based on the userID 
def createPairs(df):
    
    moviePairs = df.alias("ratings1") \
      .join(df.alias("ratings2"), (func.col("ratings1.userId") == func.col("ratings2.userId")) \
            & (func.col("ratings1.movieId") < func.col("ratings2.movieId"))) \
      .select(func.col("ratings1.movieId").alias("movie1"), \
        func.col("ratings2.movieId").alias("movie2"), \
        func.col("ratings1.rating").alias("rating1"), \
        func.col("ratings2.rating").alias("rating2")).cache()
    
    return moviePairs

##calculate the similarity score for a pair of movies using Pearson correlation coefficient
##higher correlation coefficient means the movies are more similar based on the users ratings of both movies     
def calculateSimilarities(df):
    
    movieSims = df.groupBy("movie1","movie2") \
        .agg(func.count("movie1").alias("count"),func.corr(func.col("rating1"), func.col("rating2")).alias("score")).cache()
      
    return movieSims    
      

##this function turns a moviePair dataframe to a dataframe of movieIDs and movieIDs that are similar to them, including the similarity score calcuated before        
def organizeSimilarities(df,min_count, min_score, num_results):
    
    temp1_df = df.filter((func.col("count") > min_count) & (func.col("score") > min_score)).select(func.col("movie1").alias("movieId"),func.col("movie2").alias("recID"), func.col("score"), func.col("count"))
    
    temp2_df = df.filter((func.col("count") > min_count) & (func.col("score") > min_score)).select(func.col("movie2").alias("movieId"),func.col("movie1").alias("recID"), func.col("score"),func.col("count"))
    
    union_df = temp1_df.union(temp2_df)
    
    ##movie1_windowSpec = Window.partitionBy("movieId").orderBy(func.col("score").desc())
    
    ##alternative windowing function where we order the movies by the product of score and count
    
    ##movie1_windowSpec = Window.partitionBy("movieId").orderBy((func.col("score") * func.col("count")).desc())
    
    ##another alternative window: the order of the movies is a product of log of count column and score
    ##this might produce better similarity results since it accounts for rating score and normalized number of reviewers 
    
    ##order the similar movies for each movie using a window function, similarity score and number of ratings for each pair of movies 
    ##using a log function to scale the number of ratings 
    movie1_windowSpec = Window.partitionBy("movieId").orderBy((func.col("score") * func.log(func.col("count"))).desc())

    ##return a dataframe with at most 10 similar movies with the highest similarity score 
    rec_df = union_df.withColumn("score_number",func.row_number().over(movie1_windowSpec)).filter(func.col("score_number")< num_results+1).cache()
    
    
    return rec_df


## main program

if __name__ == "__main__":

    ## reading and writing from S3
    aws_bucket_name = "mj-databricks-data"

    ## S3 location for EMR
    file_location_movies = "s3://mjendrisek-emr/data/ml-25m/ml-25m/movies.csv"
    file_location_ratings = "s3://mjendrisek-emr/data/ml-25m/ml-25m/ratings.csv"

    file_tag = "10-25NormalizedCount"

    write_file_location = "s3://mjendrisek-emr/movieRecommendations/" + file_tag

    ## this is needed if you run this script on EMR
    spark = SparkSession.builder.appName("MovieRecommender").getOrCreate()

    ##create a movie dataframe
    movies_df = load_dfs(file_location_movies, "csv", True, True, ",")


    ##create a ratings dataframe 
    ratings_df = load_dfs(file_location_ratings, "csv", True, True, ",")

    ##create movie pairs for movies rated by the same user
    moviePairs = createPairs(ratings_df)

    ##calculate movie pair similarities between the pair of movies rated by the same user
    ##the main idea is that if the same movie pair rated by the same user has higher similarity score, they can be recommended as a similar movie
    similarMovies = calculateSimilarities(moviePairs)

    ## create a list of similar movies with a threshold  
    similarMovies_df = organizeSimilarities(similarMovies, 129, 0.35, 10)

    ##create a dataframe of similar movies for each movie: movieID -> 10 movieIDs representing the most similar movies 
    rec_10_df = similarMovies_df.groupBy("movieId").agg(func.collect_list("recID")).cache()
    
    ##return a list of similar movies from a dataframe
    rec_10_list = rec_10_df.collect()

    recommendations_list = []

    ##populate a list of recommended movies; first item in the list is a movieID associated a movie associated with the recommendations, following movieIDs are recommended movies 
    for rec in rec_10_list:
        recommendations =[]
        recommendations = rec[1]
        ##if there are less than 10 recommended movies, populate the rest with None so the list can be saved as a file later
        recommendations.insert(0,rec[0])
           while (len(recommendations) <11 ):
                recommendations.append(None)
    
    recommendations_list.append(recommendations)
 
    ##create a dataframe of recommended movies and save it as csv file to be used in the recommender service 
    columns = ["movieID", "rec1", "rec2","rec3","rec4","rec5","rec6","rec7","rec8","rec9","rec10"]
    rec_final_df = spark.createDataFrame(recommendations_list).toDF(*columns)

    rec_final_df.coalesce(1).write.csv(write_file_location,"overwrite")
