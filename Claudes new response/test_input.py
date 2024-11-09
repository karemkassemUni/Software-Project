def test_input(filename):
    try:
        with open(filename, 'r') as f:
            print("File opened successfully")
            points = []
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        row = [float(x) for x in line.split(',')]
                        points.append(row)
                        print(f"Line {i}: {row}")
                    except ValueError as e:
                        print(f"Error parsing line {i}: {e}")
                        
            if points:
                n = len(points)
                d = len(points[0])
                print(f"\nSummary:")
                print(f"Number of points (n): {n}")
                print(f"Dimensions (d): {d}")
                print(f"First point: {points[0]}")
                print(f"Last point: {points[-1]}")
                
                # Check consistency
                consistent = all(len(row) == d for row in points)
                print(f"Dimensions consistent: {consistent}")
            else:
                print("No valid points found in file")
                
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python test_input.py <input_file>")
    else:
        test_input(sys.argv[1])