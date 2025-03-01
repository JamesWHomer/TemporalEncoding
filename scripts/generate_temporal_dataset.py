#!/usr/bin/env python3
# Copyright 2024
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generate a synthetic dataset for training GemmaTE on temporal reasoning tasks.
This script creates examples of users asking about dates/times and appropriate responses.
"""

import argparse
import json
import os
import random
from datetime import datetime, timedelta
import time
from typing import List, Dict, Tuple, Any

# Define May 23, 2006 as the reference point
REF_DATE = datetime(2006, 5, 23)
# Maximum date (100 years in the future)
MAX_DATE = REF_DATE + timedelta(days=365 * 100)
# Define date formats
DATE_FORMATS = [
    "%B %d, %Y",           # January 1, 2023
    "%b %d, %Y",           # Jan 1, 2023
    "%d %B %Y",            # 1 January 2023
    "%d %b %Y",            # 1 Jan 2023
    "%Y-%m-%d",            # 2023-01-01
    "%m/%d/%Y",            # 01/01/2023
    "%d/%m/%Y",            # 01/01/2023 (European style)
    "%A, %B %d, %Y",       # Monday, January 1, 2023
    "%A, %d %B %Y",        # Monday, 1 January 2023
]

# Define time formats
TIME_FORMATS = [
    "%I:%M %p",            # 12:30 PM
    "%H:%M",               # 14:30
    "%I:%M:%S %p",         # 12:30:45 PM
    "%H:%M:%S",            # 14:30:45
]

# Define datetime formats (combining date and time)
DATETIME_FORMATS = [
    "%B %d, %Y at %I:%M %p",        # January 1, 2023 at 12:30 PM
    "%A, %B %d, %Y, %I:%M %p",      # Monday, January 1, 2023, 12:30 PM
    "%d %b %Y, %H:%M",              # 1 Jan 2023, 14:30
    "%Y-%m-%d %H:%M:%S",            # 2023-01-01 14:30:45
]

# Question templates about dates
DATE_QUESTIONS = [
    "What is today's date?",
    "What day is it today?",
    "What's the date today?",
    "Could you tell me today's date?",
    "What is the date today?",
    "What date is it?",
    "Do you know what day it is?",
    "Can you tell me what date it is?",
    "I need to know today's date, can you help?",
    "What's today's date?",
    "What day of the month is it?",
    "What day of the week is it?",
    "What's the current date?",
    "Tell me the date.",
    "Tell me what day it is today.",
    "What day and date is it?",
    "I forgot what date it is today.",
    "What's the date?",
    "What is the current date?",
    "What date is it today?",
]

# Question templates about time
TIME_QUESTIONS = [
    "What time is it?",
    "Could you tell me the current time?",
    "Do you know what time it is?",
    "What's the time right now?",
    "Can you tell me the time?",
    "What is the current time?",
    "What time do we have now?",
    "Tell me the time, please.",
    "I need to know the current time.",
    "What's the current time?",
    "What's the time?",
    "What time do we have?",
    "Can you check what time it is?",
    "Tell me the current time.",
    "I'm wondering what time it is.",
    "Do you have the time?",
    "Would you tell me what time it is?",
    "What hour is it?",
    "Could you check the time for me?",
    "I need the current time.",
]

# Question templates about both date and time
DATETIME_QUESTIONS = [
    "What's the date and time?",
    "Can you tell me the date and time?",
    "What is the current date and time?",
    "Could you tell me the date and time right now?",
    "What date and time is it?",
    "I need to know the current date and time.",
    "What day and time is it?",
    "Tell me today's date and the current time.",
    "What's today's date and time?",
    "Do you know the current date and time?",
    "Can you provide me with the date and time?",
    "What's the exact date and time now?",
    "I'd like to know the current date and time.",
    "What day of the week is it and what time?",
    "Could you share the current date and time?",
    "Give me the date and time please.",
    "I need the current date and time.",
    "Tell me what date and time it is.",
    "What date and time do we have now?",
    "What's the current date and time?",
]

# Questions about specific components of date/time
SPECIFIC_DATE_QUESTIONS = [
    "What month is it?",
    "What year is it?",
    "What day of the week is it?",
    "What's the current month?",
    "Which month are we in?",
    "What year are we in?",
    "What's the current year?",
    "Which day of the week is today?",
    "Is it morning or afternoon?",
    "What quarter of the year is it?",
    "What season is it?",
    "Are we in daylight saving time?",
    "How many days are left in this month?",
    "How many days until the end of the year?",
    "What week of the year is it?",
]

# Questions about relative dates
RELATIVE_DATE_QUESTIONS = [
    "What day was yesterday?",
    "What will be the date tomorrow?",
    "What was the date last week?",
    "What will be the date next week?",
    "What date was it 10 days ago?",
    "What date will it be in 2 weeks?",
    "What month was it 3 months ago?",
    "What month will it be in 2 months?",
    "What year was it 5 years ago?",
    "What year will it be in 10 years?",
    "What day of the week was 3 days ago?",
    "What day of the week will it be in 4 days?",
    "What was the date last Monday?",
    "What will be the date next Friday?",
    "Was yesterday a weekday or weekend?",
    "Will tomorrow be a weekday or weekend?",
]

# Special date questions
SPECIAL_DATE_QUESTIONS = [
    "When is the next New Year's Day?",
    "When is Christmas this year?",
    "When is the next Valentine's Day?",
    "When is Halloween this year?",
    "When is Thanksgiving this year?",
    "When is the next Fourth of July?",
    "When is Labor Day this year?",
    "When is the next Memorial Day?",
    "When is Easter this year?",
    "When is the next Earth Day?",
    "When is the next daylight saving time change?",
    "When is the next leap year?",
    "When is the start of summer this year?",
    "When is the winter solstice this year?",
    "When is the next Friday the 13th?",
]

# Date response templates
DATE_RESPONSES = [
    "Today is {date}.",
    "The date today is {date}.",
    "It's {date}.",
    "Today's date is {date}.",
    "The current date is {date}.",
    "Right now, it's {date}.",
    "It is {date} today.",
    "As of now, it's {date}.",
    "Currently, the date is {date}.",
    "It's currently {date}.",
]

# Time response templates
TIME_RESPONSES = [
    "The time is {time}.",
    "It's {time}.",
    "The current time is {time}.",
    "Right now, it's {time}.",
    "It is {time} right now.",
    "The time right now is {time}.",
    "At the moment, it's {time}.",
    "Currently, the time is {time}.",
    "It's currently {time}.",
    "As of now, it's {time}.",
]

# DateTime response templates
DATETIME_RESPONSES = [
    "It's {datetime}.",
    "The current date and time is {datetime}.",
    "Right now, it's {datetime}.",
    "It is {datetime}.",
    "The date and time is {datetime}.",
    "Currently, it's {datetime}.",
    "As of now, it's {datetime}.",
    "The current date and time are {datetime}.",
    "Today is {datetime}.",
    "It's currently {datetime}.",
]

# Month response templates
MONTH_RESPONSES = [
    "It's {month}.",
    "The current month is {month}.",
    "We're in {month}.",
    "The month is {month}.",
    "It's {month} right now.",
    "Currently, it's {month}.",
    "We are in the month of {month}.",
    "Right now, it's {month}.",
    "The month is currently {month}.",
    "As of now, it's {month}.",
]

# Year response templates
YEAR_RESPONSES = [
    "It's {year}.",
    "The current year is {year}.",
    "We're in {year}.",
    "The year is {year}.",
    "It's {year} right now.",
    "Currently, it's {year}.",
    "We are in the year {year}.",
    "Right now, it's {year}.",
    "The year is currently {year}.",
    "As of now, it's {year}.",
]

# Day of week response templates
DAY_OF_WEEK_RESPONSES = [
    "Today is {day_of_week}.",
    "It's {day_of_week}.",
    "The day is {day_of_week}.",
    "Today's day is {day_of_week}.",
    "Currently, it's {day_of_week}.",
    "Right now, it's {day_of_week}.",
    "The current day is {day_of_week}.",
    "As of now, it's {day_of_week}.",
    "It's {day_of_week} today.",
    "The day of the week is {day_of_week}.",
]

# Relative date response templates
RELATIVE_DATE_RESPONSES = [
    "{relative_query} was/will be {relative_date}.",
    "The date {relative_query} was/will be {relative_date}.",
    "{relative_query}, it was/will be {relative_date}.",
    "{relative_query} is/was {relative_date}.",
    "It was/will be {relative_date} {relative_query}.",
    "The date {relative_query} is/was {relative_date}.",
    "{relative_query} falls/fell on {relative_date}.",
    "{relative_date} is/was the date {relative_query}.",
    "{relative_query}, the date was/will be {relative_date}.",
    "For {relative_query}, it's/it was {relative_date}.",
]

# Special date response templates
SPECIAL_DATE_RESPONSES = [
    "{special_date} is on {special_date_answer}.",
    "{special_date} falls on {special_date_answer}.",
    "{special_date} will be on {special_date_answer}.",
    "This year, {special_date} is on {special_date_answer}.",
    "{special_date} is coming up on {special_date_answer}.",
    "Mark your calendar: {special_date} is on {special_date_answer}.",
    "{special_date} will be celebrated on {special_date_answer}.",
    "{special_date} happens on {special_date_answer}.",
    "You can expect {special_date} on {special_date_answer}.",
    "{special_date} is scheduled for {special_date_answer}.",
]

def generate_random_datetime(start: datetime = REF_DATE, end: datetime = MAX_DATE) -> datetime:
    """Generate a random datetime between start and end"""
    delta = end - start
    int_delta = delta.days * 24 * 60 * 60 + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)

def format_datetime(dt: datetime, format_str: str) -> str:
    """Format datetime according to the given format string"""
    return dt.strftime(format_str)

def generate_date_example(dt: datetime) -> Dict[str, str]:
    """Generate a date question and answer pair"""
    question = random.choice(DATE_QUESTIONS)
    response_template = random.choice(DATE_RESPONSES)
    date_format = random.choice(DATE_FORMATS)
    formatted_date = format_datetime(dt, date_format)
    response = response_template.format(date=formatted_date)
    
    return {
        "prompt": question,
        "response": response,
        "timestamp_ms": int(dt.timestamp() * 1000)
    }

def generate_time_example(dt: datetime) -> Dict[str, str]:
    """Generate a time question and answer pair"""
    question = random.choice(TIME_QUESTIONS)
    response_template = random.choice(TIME_RESPONSES)
    time_format = random.choice(TIME_FORMATS)
    formatted_time = format_datetime(dt, time_format)
    response = response_template.format(time=formatted_time)
    
    return {
        "prompt": question,
        "response": response,
        "timestamp_ms": int(dt.timestamp() * 1000)
    }

def generate_datetime_example(dt: datetime) -> Dict[str, str]:
    """Generate a datetime question and answer pair"""
    question = random.choice(DATETIME_QUESTIONS)
    response_template = random.choice(DATETIME_RESPONSES)
    datetime_format = random.choice(DATETIME_FORMATS)
    formatted_datetime = format_datetime(dt, datetime_format)
    response = response_template.format(datetime=formatted_datetime)
    
    return {
        "prompt": question,
        "response": response,
        "timestamp_ms": int(dt.timestamp() * 1000)
    }

def generate_month_example(dt: datetime) -> Dict[str, str]:
    """Generate a month question and answer pair"""
    question = "What month is it?"
    response_template = random.choice(MONTH_RESPONSES)
    month = dt.strftime("%B")
    response = response_template.format(month=month)
    
    return {
        "prompt": question,
        "response": response,
        "timestamp_ms": int(dt.timestamp() * 1000)
    }

def generate_year_example(dt: datetime) -> Dict[str, str]:
    """Generate a year question and answer pair"""
    question = "What year is it?"
    response_template = random.choice(YEAR_RESPONSES)
    year = dt.strftime("%Y")
    response = response_template.format(year=year)
    
    return {
        "prompt": question,
        "response": response,
        "timestamp_ms": int(dt.timestamp() * 1000)
    }

def generate_day_of_week_example(dt: datetime) -> Dict[str, str]:
    """Generate a day of week question and answer pair"""
    question = "What day of the week is it?"
    response_template = random.choice(DAY_OF_WEEK_RESPONSES)
    day_of_week = dt.strftime("%A")
    response = response_template.format(day_of_week=day_of_week)
    
    return {
        "prompt": question,
        "response": response,
        "timestamp_ms": int(dt.timestamp() * 1000)
    }

def generate_relative_date_example(dt: datetime) -> Dict[str, str]:
    """Generate a relative date question and answer pair"""
    # Define relative date options (past and future)
    relative_options = [
        ("yesterday", dt - timedelta(days=1)),
        ("tomorrow", dt + timedelta(days=1)),
        ("last week", dt - timedelta(days=7)),
        ("next week", dt + timedelta(days=7)),
        ("a week ago", dt - timedelta(days=7)),
        ("in a week", dt + timedelta(days=7)),
        ("last month", dt - timedelta(days=30)),
        ("next month", dt + timedelta(days=30)),
        ("3 days ago", dt - timedelta(days=3)),
        ("in 3 days", dt + timedelta(days=3)),
        ("last year", dt - timedelta(days=365)),
        ("next year", dt + timedelta(days=365)),
    ]
    
    # Choose a random relative option
    relative_query, relative_dt = random.choice(relative_options)
    
    # Customize question based on tense
    if "ago" in relative_query or "last" in relative_query or "yesterday" == relative_query:
        question = f"What date was {relative_query}?"
        relative_verb = "was"
    else:
        question = f"What date will be {relative_query}?"
        relative_verb = "will be"
    
    # Format response
    date_format = random.choice(DATE_FORMATS)
    formatted_date = format_datetime(relative_dt, date_format)
    response = f"The date {relative_query} {relative_verb} {formatted_date}."
    
    return {
        "prompt": question,
        "response": response,
        "timestamp_ms": int(dt.timestamp() * 1000)
    }

def generate_special_date_example(dt: datetime) -> Dict[str, str]:
    """Generate a special date question and answer pair"""
    year = dt.year
    
    # Define special dates
    special_dates = {
        "New Year's Day": datetime(year, 1, 1),
        "Valentine's Day": datetime(year, 2, 14),
        "Earth Day": datetime(year, 4, 22),
        "Fourth of July": datetime(year, 7, 4),
        "Halloween": datetime(year, 10, 31),
        "Christmas": datetime(year, 12, 25),
        "New Year's Eve": datetime(year, 12, 31),
    }
    
    # Choose a random special date
    special_date_name, special_date_dt = random.choice(list(special_dates.items()))
    
    # If the special date has already passed this year, use next year
    if special_date_dt < dt:
        special_date_dt = datetime(year + 1, special_date_dt.month, special_date_dt.day)
        question = f"When is the next {special_date_name}?"
    else:
        question = f"When is {special_date_name} this year?"
    
    # Format response
    date_format = random.choice(["%B %d, %Y", "%A, %B %d, %Y"])
    formatted_date = format_datetime(special_date_dt, date_format)
    response = f"{special_date_name} is on {formatted_date}."
    
    return {
        "prompt": question,
        "response": response,
        "timestamp_ms": int(dt.timestamp() * 1000)
    }

def generate_dataset(num_examples: int, output_file: str) -> None:
    """Generate the dataset with the specified number of examples"""
    examples = []
    generation_methods = [
        generate_date_example,
        generate_time_example,
        generate_datetime_example,
        generate_month_example,
        generate_year_example,
        generate_day_of_week_example,
        generate_relative_date_example,
        generate_special_date_example,
    ]
    
    # Assign weights to make some question types more common
    weights = [3, 3, 3, 1, 1, 1, 2, 1]
    
    # Print progress update
    print(f"Generating {num_examples} examples...")
    
    # Generate examples
    for i in range(num_examples):
        # Generate a random datetime
        dt = generate_random_datetime()
        
        # Choose a generation method based on weights
        generation_method = random.choices(generation_methods, weights=weights, k=1)[0]
        
        # Generate an example
        example = generation_method(dt)
        examples.append(example)
        
        # Print progress update
        if (i + 1) % 10000 == 0:
            print(f"Generated {i + 1} examples")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write examples to file
    with open(output_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Dataset generation complete! {num_examples} examples written to {output_file}")
    
    # Print a sample of 5 examples
    print("\nSample examples:")
    for i in range(min(5, len(examples))):
        example = examples[i]
        dt = datetime.fromtimestamp(example["timestamp_ms"] / 1000)
        print(f"Example {i+1}:")
        print(f"  Date: {dt}")
        print(f"  Prompt: {example['prompt']}")
        print(f"  Response: {example['response']}")
        print(f"  Timestamp (ms): {example['timestamp_ms']}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset for temporal encoding training")
    parser.add_argument("--num_examples", type=int, default=100000, help="Number of examples to generate")
    parser.add_argument("--output_file", type=str, default="data/temporal_dataset.jsonl", help="Output file path")
    args = parser.parse_args()
    
    generate_dataset(args.num_examples, args.output_file)

if __name__ == "__main__":
    main() 