ORIGINAL_EXTRA_BOTTOM = 2;
NEW_EXTRA_BOTTOM = 27.92 - 3;


// and we also need to account for the mount thickness which is 7.9mm
DIFF_NEEDED = NEW_EXTRA_BOTTOM - ORIGINAL_EXTRA_BOTTOM + 7.9;

echo("total diff:");
echo(DIFF_NEEDED);


$fn = 90;

// make a cylinder that is DIFF_NEEDED tall and 80mm diameter

translate([0, 0, -NEW_EXTRA_BOTTOM - 7.9])
  cylinder(h=DIFF_NEEDED, r=80/2);