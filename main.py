import input_files
import venus
import venus_climate
import earth
import earth_climate
import mars
# import mars_climate
import titan
import titan_climate

def main():
    input_files.main()
    venus.main()
    venus_climate.main()
    earth.main()
    earth_climate.main()
    mars.main()
    titan.main()
    titan_climate.main()
    
if __name__ == '__main__':
    main()