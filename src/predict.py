import argparse
import os

from src.predictor import predict


def print_result(result):

    print("\n")

    print("=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print(f"Category       : {result['category']}")
    print(f"Confidence     : {result['confidence']} %")
    print(f"Weight         : {result['weight']} Kg")
    print(f"Carbon Saved   : {result['carbon_saved']} Kg CO2e")
    print(f"Points         : {result['points']}")
    print(f"Eco Score      : {result['eco_score']}")

    print("=" * 60)


def main():

    parser = argparse.ArgumentParser(

        description="EcoScan AI Prediction"

    )

    parser.add_argument(

        "image",

        help="Image file path"

    )

    parser.add_argument(

        "--weight",

        type=float,

        default=1.0,

        help="Waste Weight (Kg)"

    )

    args = parser.parse_args()

    if not os.path.exists(args.image):

        print("\n")

        print("=" * 60)
        print("ERROR")
        print("=" * 60)
        print("Image not found :")
        print(args.image)
        print("=" * 60)

        return

    result = predict(

        image_path=args.image,

        weight=args.weight

    )

    print_result(result)


if __name__ == "__main__":

    main()