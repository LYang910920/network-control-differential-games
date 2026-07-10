"""Compatibility imports for named foundation experiment profiles."""

from cybercontrol.foundation_profiles import *  # noqa: F403
from cybercontrol.foundation_profiles import describe_profiles


if __name__ == "__main__":
    for row in describe_profiles():
        print(
            f"{row['name']}: type={row['type']}, horizon={row['horizon']}, "
            f"{row['rates']}, {row['bounds']}, {row['notes']}"
        )
