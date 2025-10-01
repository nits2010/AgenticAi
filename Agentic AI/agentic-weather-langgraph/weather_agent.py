"""Weather agent main entry point"""

import argparse
import sys
import os

# Load environment variables
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    # Fallback: simple .env loader if python-dotenv not available
    import pathlib
    def load_dotenv():
        env_path = pathlib.Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")
    load_dotenv()

from agent import WeatherAgent
from config import config


def run_demo():
    """Run demo queries"""
    agent = WeatherAgent()
    
    demo_queries = [
        "Weather in Mumbai for 3 days",
        "Any rain alerts for Bengaluru tomorrow?",
        "12.97,77.59 for 2 days in fahrenheit"
    ]
    
    print("🧠 Weather Agent Demo")
    print("=" * 50)
    
    for query in demo_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 30)
        try:
            result = agent.run(query)
            print(result["answer"])
        except Exception as e:
            print(f"❌ Error: {e}")
        print()


def run_tests():
    """Run unit tests"""
    import unittest
    
    class WeatherAgentTests(unittest.TestCase):
        def test_query_parser(self):
            from agent import QueryParser
            
            # Test coordinate parsing
            location, days, units = QueryParser.parse("19.0760,72.8777 for 2 days")
            self.assertEqual(location, "19.0760,72.8777")
            self.assertEqual(days, 2)
            self.assertEqual(units, "metric")
            
            # Test location parsing
            location, days, units = QueryParser.parse("Weather in Mumbai for 3 days")
            self.assertEqual(location, "mumbai")
            self.assertEqual(days, 3)
            self.assertEqual(units, "metric")
        
        def test_config_validation(self):
            # Test that config is valid
            config.validate()
    
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(WeatherAgentTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Simplified Weather Agent")
    parser.add_argument("--query", "-q", help="Weather query to process")
    parser.add_argument("--feedback", "-f", help="Feedback for the agent")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--provider", "-p", choices=["auto", "groq", "openai"], default="auto", 
                       help="LLM provider to use (default: auto-detect)")
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
        return
    
    if args.query:
        try:
            agent = WeatherAgent(llm_provider=args.provider)
            result = agent.run(args.query, feedback=args.feedback)
            print(result["answer"])
        except Exception as e:
            print(f"Error: {e}")
            print("\n💡 Tips:")
            print("- For Groq: Set GROQ_API_KEY in .env file")
            print("- For OpenAI: Set OPENAI_API_KEY in .env file")
            print("- Get Groq API key: https://console.groq.com/keys")
            print("- Get OpenAI API key: https://platform.openai.com/api-keys")
    else:
        run_demo()


if __name__ == "__main__":
    main()
