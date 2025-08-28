#!/usr/bin/env python3
"""
Database migration to add sentiment analysis fields to facts table.
This migration is optional - the system will work without these fields.
"""

import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from config.settings import DATABASE_CONFIG
from storage.db_utils import get_conn


def add_sentiment_fields():
    """
    Add sentiment analysis fields to the facts table.
    These fields are optional and will be computed in-memory if not present.
    """
    db_path = DATABASE_CONFIG["default_path"]

    try:
        with get_conn(db_path) as conn:
            # Check if fields already exist
            cursor = conn.execute("PRAGMA table_info(facts)")
            existing_columns = [row[1] for row in cursor.fetchall()]

            # Fields to add
            new_fields = [
                ("affinity_score", "FLOAT"),
                ("sentiment_slope", "FLOAT"),
                ("volatility_score", "FLOAT"),
            ]

            added_fields = []
            for field_name, field_type in new_fields:
                if field_name not in existing_columns:
                    try:
                        conn.execute(
                            f"ALTER TABLE facts ADD COLUMN {field_name} {field_type}"
                        )
                        added_fields.append(field_name)
                        print(f"✅ Added column: {field_name}")
                    except Exception as e:
                        print(f"❌ Error adding {field_name}: {e}")
                else:
                    print(f"ℹ️ Column {field_name} already exists")

            if added_fields:
                print(
                    f"\n🎉 Successfully added {len(added_fields)} sentiment fields to facts table"
                )
                print(
                    "   These fields will be populated automatically as facts are processed"
                )
            else:
                print("\nℹ️ All sentiment fields already exist in the database")

            # Commit changes
            conn.commit()

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

    return True


def populate_sentiment_fields():
    """
    Populate sentiment fields for existing facts.
    This is a one-time operation to backfill existing data.
    """
    from storage.memory_log import MemoryLog
    from storage.memory_utils import (get_sentiment_score,
                                      get_sentiment_trajectory,
                                      get_volatility_score)

    print("\n🔄 Populating sentiment fields for existing facts...")

    try:
        memory_log = MemoryLog()
        all_facts = memory_log.get_all_facts(prune_contradictions=False)

        if not all_facts:
            print("ℹ️ No facts found to populate")
            return True

        print(f"📊 Processing {len(all_facts)} facts...")

        # Group facts by subject for trajectory analysis
        subject_groups = {}
        for fact in all_facts:
            subject = fact.subject.lower().strip()
            if subject not in subject_groups:
                subject_groups[subject] = []
            subject_groups[subject].append(fact)

        # Calculate sentiment scores and update database
        updated_count = 0
        with get_conn(memory_log.db_path) as conn:
            for fact in all_facts:
                try:
                    # Calculate affinity score
                    affinity_score = get_sentiment_score(fact.predicate)

                    # Get trajectory for this subject
                    subject_facts = subject_groups.get(fact.subject.lower().strip(), [])
                    trajectory = get_sentiment_trajectory(subject_facts)
                    sentiment_slope = trajectory["slope"]
                    volatility_score = trajectory["volatility"]

                    # Update the fact
                    conn.execute(
                        """
                        UPDATE facts 
                        SET affinity_score = ?, sentiment_slope = ?, volatility_score = ?
                        WHERE id = ?
                    """,
                        (affinity_score, sentiment_slope, volatility_score, fact.id),
                    )

                    updated_count += 1

                    if updated_count % 10 == 0:
                        print(f"   Processed {updated_count}/{len(all_facts)} facts...")

                except Exception as e:
                    print(f"⚠️ Error processing fact {fact.id}: {e}")
                    continue

            conn.commit()

        print(f"✅ Successfully updated {updated_count} facts with sentiment data")
        return True

    except Exception as e:
        print(f"❌ Error populating sentiment fields: {e}")
        return False


def main():
    """Run the migration"""
    print("🧠 MeRNSTA Sentiment Analysis Database Migration")
    print("=" * 60)

    # Add new fields
    if add_sentiment_fields():
        print("\n📊 Migration completed successfully!")

        # Ask if user wants to populate existing data
        try:
            response = (
                input("\nPopulate sentiment fields for existing facts? (y/N): ")
                .strip()
                .lower()
            )
            if response in ["y", "yes"]:
                populate_sentiment_fields()
        except (EOFError, KeyboardInterrupt):
            print("\nSkipping data population")
    else:
        print("\n❌ Migration failed!")


if __name__ == "__main__":
    main()
