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
Generate a synthetic dataset of multi-turn conversations for training GemmaTE.
This script creates examples of conversations focused on temporal topics.
"""

import argparse
import json
import os
import random
from datetime import datetime, timedelta
import time
from typing import List, Dict, Any

# Define May 23, 2006 as the reference point
REF_DATE = datetime(2006, 5, 23)
# Maximum date (100 years in the future)
MAX_DATE = REF_DATE + timedelta(days=365 * 100)

# Define conversation templates
CONVERSATION_TEMPLATES = [
    # Template 1: Basic date and time questioning
    [
        {"role": "user", "template": "What's today's date?"},
        {"role": "assistant", "template": "Today is {date_full}."},
        {"role": "user", "template": "And what time is it right now?"},
        {"role": "assistant", "template": "The current time is {time}."},
        {"role": "user", "template": "Thanks! What day of the week is it?"},
        {"role": "assistant", "template": "It's {day_of_week}."}
    ],
    
    # Template 2: Planning a future event
    [
        {"role": "user", "template": "I'm planning an event for {future_date_rel}. What day of the week will that be?"},
        {"role": "assistant", "template": "{future_date_full} will be a {future_day_of_week}."},
        {"role": "user", "template": "Perfect. And if I reschedule to exactly one week later?"},
        {"role": "assistant", "template": "One week later would be {future_date_plus_week_full}, which is also a {future_day_of_week}."},
        {"role": "user", "template": "Great, thanks!"},
        {"role": "assistant", "template": "You're welcome! Let me know if you need any other date calculations for your planning."}
    ],
    
    # Template 3: Birthday calculation
    [
        {"role": "user", "template": "My birthday is on {birthday_date_short}. How many days are there until my next birthday?"},
        {"role": "assistant", "template": "Today is {date_full}. Your next birthday on {next_birthday_full} is {days_to_birthday} days away."},
        {"role": "user", "template": "How old will I be then?"},
        {"role": "assistant", "template": "Assuming you were born on {birthday_date_full}, you will be {next_age} years old on your next birthday."},
        {"role": "user", "template": "And what day of the week will my birthday fall on?"},
        {"role": "assistant", "template": "Your birthday on {next_birthday_full} will fall on a {next_birthday_day_of_week}."}
    ],
    
    # Template 4: Historical date
    [
        {"role": "user", "template": "What was the date {historical_time_ago}?"},
        {"role": "assistant", "template": "{historical_time_ago} was {historical_date_full}."},
        {"role": "user", "template": "And what weekday was that?"},
        {"role": "assistant", "template": "It was a {historical_day_of_week}."},
        {"role": "user", "template": "How many days ago was that exactly?"},
        {"role": "assistant", "template": "From today ({date_full}), {historical_date_full} was exactly {days_since_historical} days ago."}
    ],
    
    # Template 5: Time conversion
    [
        {"role": "user", "template": "What time is it now?"},
        {"role": "assistant", "template": "It's currently {time} on {date_full}."},
        {"role": "user", "template": "What would that be in GMT?"},
        {"role": "assistant", "template": "Assuming we're in {current_timezone}, the current time in GMT would be {time_gmt}."},
        {"role": "user", "template": "And what about in Tokyo?"},
        {"role": "assistant", "template": "The current time in Tokyo (JST) would be {time_tokyo}."}
    ],
    
    # Template 6: Holiday planning
    [
        {"role": "user", "template": "When is {holiday} this year?"},
        {"role": "assistant", "template": "{holiday} {holiday_year_verb} on {holiday_date_full} this year."},
        {"role": "user", "template": "How many days away is that?"},
        {"role": "assistant", "template": "Today is {date_full}. {holiday} is {days_to_holiday} days away."},
        {"role": "user", "template": "And next year?"},
        {"role": "assistant", "template": "Next year, {holiday} will be on {next_year_holiday_date_full}, which is a {next_year_holiday_day_of_week}."}
    ],
    
    # Template 7: Calendar calculation
    [
        {"role": "user", "template": "Today is {date_full}. If I have a meeting every 3 days, when will the next 3 meetings be?"},
        {"role": "assistant", "template": "Starting from today ({date_full}), your next three meetings will be on:\n1. {meeting1_date_full} ({meeting1_day_of_week})\n2. {meeting2_date_full} ({meeting2_day_of_week})\n3. {meeting3_date_full} ({meeting3_day_of_week})"},
        {"role": "user", "template": "What if I change it to every 5 days instead?"},
        {"role": "assistant", "template": "If you change to every 5 days, starting from today ({date_full}), your next three meetings will be on:\n1. {alt_meeting1_date_full} ({alt_meeting1_day_of_week})\n2. {alt_meeting2_date_full} ({alt_meeting2_day_of_week})\n3. {alt_meeting3_date_full} ({alt_meeting3_day_of_week})"}
    ],
    
    # Template 8: Season inquiry
    [
        {"role": "user", "template": "What season is it currently?"},
        {"role": "assistant", "template": "Today is {date_full}. In the Northern Hemisphere, it's currently {current_season_north}. In the Southern Hemisphere, it's {current_season_south}."},
        {"role": "user", "template": "When does the next season start?"},
        {"role": "assistant", "template": "In the Northern Hemisphere, {next_season_north} begins on {next_season_date_north_full}. In the Southern Hemisphere, {next_season_south} begins on that same date."},
        {"role": "user", "template": "How many days until then?"},
        {"role": "assistant", "template": "There are {days_to_next_season} days until the next season change."}
    ],
    
    # Template 9: Age calculation
    [
        {"role": "user", "template": "If someone was born on {birth_date_full}, how old are they today?"},
        {"role": "assistant", "template": "Today is {date_full}. Someone born on {birth_date_full} would be {current_age} years old today."},
        {"role": "user", "template": "How many days have they been alive?"},
        {"role": "assistant", "template": "They have been alive for approximately {days_alive} days."},
        {"role": "user", "template": "And when will they turn {next_milestone_age}?"},
        {"role": "assistant", "template": "They will turn {next_milestone_age} on {milestone_date_full}, which is {days_to_milestone} days from today."}
    ],
    
    # Template 10: Countdown calculation
    [
        {"role": "user", "template": "How many days until {future_holiday}?"},
        {"role": "assistant", "template": "Today is {date_full}. {future_holiday} ({future_holiday_date_full}) is {days_to_future_holiday} days away."},
        {"role": "user", "template": "What day of the week will it be?"},
        {"role": "assistant", "template": "{future_holiday} will fall on a {future_holiday_day_of_week}."},
        {"role": "user", "template": "And how many hours until then?"},
        {"role": "assistant", "template": "There are approximately {hours_to_future_holiday} hours until {future_holiday}."}
    ]
]

# Additional templates for context variables
HOLIDAYS = [
    "New Year's Day", "Valentine's Day", "St. Patrick's Day", "Easter", "Earth Day",
    "Mother's Day", "Father's Day", "Independence Day", "Labor Day", "Halloween",
    "Thanksgiving", "Christmas", "New Year's Eve"
]

FUTURE_DATE_REL = [
    "next Friday", "two weeks from now", "next month", "March 15th", 
    "the first Saturday of next month", "three days from now", "next Tuesday",
    "the end of this month", "July 4th", "December 25th"
]

HISTORICAL_TIME_AGO = [
    "yesterday", "last week", "two weeks ago", "a month ago", "last year",
    "5 years ago", "a decade ago", "in 2010", "in the year 2000", 
    "30 days ago", "last Monday", "six months ago"
]

TIMEZONES = [
    "Eastern Time (ET)", "Pacific Time (PT)", "Central Time (CT)", 
    "Mountain Time (MT)", "Eastern European Time (EET)", "Central European Time (CET)",
    "Australian Eastern Time (AET)"
]

SEASONS_NORTH = ["Winter", "Spring", "Summer", "Fall"]

def format_date(dt, format_type="full"):
    """Format a date in different formats based on the format_type"""
    if format_type == "full":
        return dt.strftime("%A, %B %d, %Y")
    elif format_type == "short":
        return dt.strftime("%m/%d/%Y")
    elif format_type == "month_day":
        return dt.strftime("%B %d")
    else:
        return dt.strftime("%Y-%m-%d")

def generate_random_datetime(start=REF_DATE, end=MAX_DATE):
    """Generate a random datetime between start and end"""
    delta = end - start
    int_delta = delta.days * 24 * 60 * 60 + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)

def get_season(dt):
    """Get the current season based on the month in Northern Hemisphere"""
    month = dt.month
    if 3 <= month <= 5:
        return "Spring"
    elif 6 <= month <= 8:
        return "Summer"
    elif 9 <= month <= 11:
        return "Fall"
    else:
        return "Winter"

def get_opposite_season(season):
    """Get the opposite season for Southern Hemisphere"""
    seasons = {"Spring": "Fall", "Summer": "Winter", "Fall": "Spring", "Winter": "Summer"}
    return seasons[season]

def get_next_season(current_season):
    """Get the next season after the current one"""
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    current_index = seasons.index(current_season)
    next_index = (current_index + 1) % 4
    return seasons[next_index]

def get_next_season_date(dt):
    """Get the date when the next season starts"""
    year = dt.year
    month = dt.month
    
    # Approximate season start dates
    if month < 3 or month == 12:  # Winter -> Spring
        return datetime(year, 3, 20)
    elif 3 <= month < 6:  # Spring -> Summer
        return datetime(year, 6, 21)
    elif 6 <= month < 9:  # Summer -> Fall
        return datetime(year, 9, 22)
    else:  # Fall -> Winter
        return datetime(year, 12, 21)

def get_next_holiday_date(dt, holiday_name):
    """Get the next occurrence of a holiday"""
    year = dt.year
    
    holiday_dates = {
        "New Year's Day": datetime(year, 1, 1),
        "Valentine's Day": datetime(year, 2, 14),
        "St. Patrick's Day": datetime(year, 3, 17),
        "Earth Day": datetime(year, 4, 22),
        "Independence Day": datetime(year, 7, 4),
        "Halloween": datetime(year, 10, 31),
        "Thanksgiving": datetime(year, 11, 25),  # Approximate - 4th Thursday in November
        "Christmas": datetime(year, 12, 25),
        "New Year's Eve": datetime(year, 12, 31)
    }
    
    if holiday_name in holiday_dates:
        holiday_date = holiday_dates[holiday_name]
        # If the holiday has already passed this year, use next year
        if holiday_date < dt:
            if holiday_name == "New Year's Day":
                holiday_date = datetime(year + 1, 1, 1)
            elif holiday_name == "Valentine's Day":
                holiday_date = datetime(year + 1, 2, 14)
            elif holiday_name == "St. Patrick's Day":
                holiday_date = datetime(year + 1, 3, 17)
            elif holiday_name == "Earth Day":
                holiday_date = datetime(year + 1, 4, 22)
            elif holiday_name == "Independence Day":
                holiday_date = datetime(year + 1, 7, 4)
            elif holiday_name == "Halloween":
                holiday_date = datetime(year + 1, 10, 31)
            elif holiday_name == "Thanksgiving":
                holiday_date = datetime(year + 1, 11, 25)
            elif holiday_name == "Christmas":
                holiday_date = datetime(year + 1, 12, 25)
            elif holiday_name == "New Year's Eve":
                holiday_date = datetime(year + 1, 12, 31)
        return holiday_date
    else:
        # Default to Christmas if holiday not found
        holiday_date = datetime(year, 12, 25)
        if holiday_date < dt:
            holiday_date = datetime(year + 1, 12, 25)
        return holiday_date

def fill_template(template, variables):
    """Fill in a template with the given variables"""
    filled_text = template
    for key, value in variables.items():
        placeholder = "{" + key + "}"
        if placeholder in filled_text:
            filled_text = filled_text.replace(placeholder, str(value))
    return filled_text

def generate_conversation_example(template_index=None):
    """Generate a conversation example using a template"""
    # Choose a template
    if template_index is None:
        template = random.choice(CONVERSATION_TEMPLATES)
    else:
        template = CONVERSATION_TEMPLATES[template_index % len(CONVERSATION_TEMPLATES)]
    
    # Generate the context date (today's date in the conversation)
    context_date = generate_random_datetime()
    
    # Create the variables dictionary
    variables = {}
    
    # Basic date and time
    variables["date_full"] = format_date(context_date, "full")
    variables["date_short"] = format_date(context_date, "short")
    variables["time"] = context_date.strftime("%I:%M %p")
    variables["day_of_week"] = context_date.strftime("%A")
    
    # Future date (randomly chosen between 1 and 60 days in the future)
    future_days = random.randint(1, 60)
    future_date = context_date + timedelta(days=future_days)
    variables["future_date_full"] = format_date(future_date, "full")
    variables["future_date_short"] = format_date(future_date, "short")
    variables["future_day_of_week"] = future_date.strftime("%A")
    variables["future_date_rel"] = random.choice(FUTURE_DATE_REL)
    
    # Future date plus one week
    future_date_plus_week = future_date + timedelta(days=7)
    variables["future_date_plus_week_full"] = format_date(future_date_plus_week, "full")
    
    # Birthday calculation
    birth_year = context_date.year - random.randint(20, 80)
    birth_month = random.randint(1, 12)
    birth_day = random.randint(1, 28)  # Avoiding edge cases with month lengths
    birthday_date = datetime(birth_year, birth_month, birth_day)
    variables["birthday_date_full"] = format_date(birthday_date, "full")
    variables["birthday_date_short"] = birthday_date.strftime("%m/%d")
    
    # Next birthday calculation
    this_year_birthday = datetime(context_date.year, birth_month, birth_day)
    next_year_birthday = datetime(context_date.year + 1, birth_month, birth_day)
    if this_year_birthday > context_date:
        next_birthday = this_year_birthday
    else:
        next_birthday = next_year_birthday
    
    variables["next_birthday_full"] = format_date(next_birthday, "full")
    variables["next_birthday_day_of_week"] = next_birthday.strftime("%A")
    variables["days_to_birthday"] = (next_birthday - context_date).days
    variables["next_age"] = next_birthday.year - birth_year
    
    # Historical date
    variables["historical_time_ago"] = random.choice(HISTORICAL_TIME_AGO)
    historical_days_ago = random.randint(1, 3650)  # Up to ~10 years ago
    historical_date = context_date - timedelta(days=historical_days_ago)
    variables["historical_date_full"] = format_date(historical_date, "full")
    variables["historical_day_of_week"] = historical_date.strftime("%A")
    variables["days_since_historical"] = historical_days_ago
    
    # Time zones
    variables["current_timezone"] = random.choice(TIMEZONES)
    gmt_offset = random.randint(-11, 12)
    tokyo_offset = 9  # Tokyo is UTC+9
    
    # Calculate GMT time
    gmt_hour = (context_date.hour - gmt_offset) % 24
    variables["time_gmt"] = datetime(context_date.year, context_date.month, context_date.day, gmt_hour, context_date.minute).strftime("%I:%M %p")
    
    # Calculate Tokyo time
    tokyo_hour = (context_date.hour + tokyo_offset - gmt_offset) % 24
    variables["time_tokyo"] = datetime(context_date.year, context_date.month, context_date.day, tokyo_hour, context_date.minute).strftime("%I:%M %p")
    
    # Holiday planning
    holiday = random.choice(HOLIDAYS)
    variables["holiday"] = holiday
    holiday_date = get_next_holiday_date(context_date, holiday)
    variables["holiday_date_full"] = format_date(holiday_date, "full")
    variables["days_to_holiday"] = (holiday_date - context_date).days
    variables["holiday_year_verb"] = "falls" if holiday_date > context_date else "fell"
    
    # Next year's holiday
    next_year_holiday_date = datetime(holiday_date.year + 1, holiday_date.month, holiday_date.day)
    variables["next_year_holiday_date_full"] = format_date(next_year_holiday_date, "full")
    variables["next_year_holiday_day_of_week"] = next_year_holiday_date.strftime("%A")
    
    # Meeting calculations
    meeting1_date = context_date + timedelta(days=3)
    meeting2_date = context_date + timedelta(days=6)
    meeting3_date = context_date + timedelta(days=9)
    
    variables["meeting1_date_full"] = format_date(meeting1_date, "full")
    variables["meeting1_day_of_week"] = meeting1_date.strftime("%A")
    variables["meeting2_date_full"] = format_date(meeting2_date, "full")
    variables["meeting2_day_of_week"] = meeting2_date.strftime("%A")
    variables["meeting3_date_full"] = format_date(meeting3_date, "full")
    variables["meeting3_day_of_week"] = meeting3_date.strftime("%A")
    
    # Alternative meeting schedule (every 5 days)
    alt_meeting1_date = context_date + timedelta(days=5)
    alt_meeting2_date = context_date + timedelta(days=10)
    alt_meeting3_date = context_date + timedelta(days=15)
    
    variables["alt_meeting1_date_full"] = format_date(alt_meeting1_date, "full")
    variables["alt_meeting1_day_of_week"] = alt_meeting1_date.strftime("%A")
    variables["alt_meeting2_date_full"] = format_date(alt_meeting2_date, "full")
    variables["alt_meeting2_day_of_week"] = alt_meeting2_date.strftime("%A")
    variables["alt_meeting3_date_full"] = format_date(alt_meeting3_date, "full")
    variables["alt_meeting3_day_of_week"] = alt_meeting3_date.strftime("%A")
    
    # Season data
    current_season_north = get_season(context_date)
    variables["current_season_north"] = current_season_north
    variables["current_season_south"] = get_opposite_season(current_season_north)
    
    next_season_north = get_next_season(current_season_north)
    variables["next_season_north"] = next_season_north
    variables["next_season_south"] = get_opposite_season(next_season_north)
    
    next_season_date = get_next_season_date(context_date)
    variables["next_season_date_north_full"] = format_date(next_season_date, "full")
    variables["days_to_next_season"] = (next_season_date - context_date).days
    
    # Age calculation
    birth_date = context_date - timedelta(days=random.randint(365*20, 365*80))
    variables["birth_date_full"] = format_date(birth_date, "full")
    
    current_age = (context_date - birth_date).days // 365
    variables["current_age"] = current_age
    variables["days_alive"] = (context_date - birth_date).days
    
    next_milestone_age = ((current_age // 10) + 1) * 10  # Next decade birthday
    variables["next_milestone_age"] = next_milestone_age
    
    years_to_milestone = next_milestone_age - current_age
    milestone_date = datetime(birth_date.year + next_milestone_age, birth_date.month, birth_date.day)
    variables["milestone_date_full"] = format_date(milestone_date, "full")
    variables["days_to_milestone"] = (milestone_date - context_date).days
    
    # Future holiday
    future_holiday = random.choice(HOLIDAYS)
    variables["future_holiday"] = future_holiday
    future_holiday_date = get_next_holiday_date(context_date, future_holiday)
    variables["future_holiday_date_full"] = format_date(future_holiday_date, "full")
    variables["future_holiday_day_of_week"] = future_holiday_date.strftime("%A")
    variables["days_to_future_holiday"] = (future_holiday_date - context_date).days
    variables["hours_to_future_holiday"] = (future_holiday_date - context_date).days * 24
    
    # Fill in the templates
    conversation = []
    for message in template:
        filled_message = {
            "role": message["role"],
            "content": fill_template(message["template"], variables)
        }
        conversation.append(filled_message)
    
    # Return the filled conversation and timestamp
    return {
        "conversation": conversation,
        "timestamp_ms": int(context_date.timestamp() * 1000)
    }

def generate_dataset(num_examples, output_file):
    """Generate a dataset with multiple conversation examples"""
    examples = []
    
    print(f"Generating {num_examples} conversation examples...")
    
    for i in range(num_examples):
        # Generate a conversation
        example = generate_conversation_example()
        examples.append(example)
        
        # Print progress updates
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1} examples")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write examples to file
    with open(output_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Dataset generation complete! {num_examples} examples written to {output_file}")
    
    # Print a sample conversation
    print("\nSample conversation:")
    sample = random.choice(examples)
    print(f"Timestamp: {datetime.fromtimestamp(sample['timestamp_ms'] / 1000)}")
    for message in sample["conversation"]:
        print(f"[{message['role']}]: {message['content']}")

def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic conversation dataset for GemmaTE training")
    parser.add_argument("--num_examples", type=int, default=10000, help="Number of conversation examples to generate")
    parser.add_argument("--output_file", type=str, default="data/temporal_conversations.jsonl", help="Output file path")
    args = parser.parse_args()
    
    generate_dataset(args.num_examples, args.output_file)

if __name__ == "__main__":
    main() 