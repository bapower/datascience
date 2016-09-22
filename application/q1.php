<?php
$doc = JFactory::getDocument();
$doc->addStyleSheet($this->baseurl . '/media/jui/css/icomoon.css');

$T = 10;
$S = 0;
$current = 0;

$moves = [
    0 => [6, 4],
    1 => [8, 6],
    2 => [7, 9],
    3 => [4, 8],
    4 => [3, 9, 0],
    5 => [],
    6 => [1, 7, 0],
    7 => [2, 6],
    8 => [1, 3],
    9 => [4, 2]
];

function getPossibilities ($moves, $sums=[['sum'=> 6, 'lastNum' => 6],['sum' => 4, 'lastNum'=>4]])
{
    for ($x = 1; $x <= 1024; $x++) {
        foreach ($sums as $i => $sum) {
            foreach ($moves[$sum['lastNum']] as $currentNum) {
                $sums[] = ['sum' => $sum['sum']+$currentNum, 'lastNum' => $currentNum];
            }
        unset($sums[$i]);
        }
    }
    return $sums;
}




echo '<pre>';
/*$sums = getPossibilities($moves);
$sumNums = [];
$sevens = [];
$fives = [];

foreach ($sums as $sum) {
    $sumNums[] = $sum['sum'];
    if ($sum['sum']%7 == 0) {
        $sevens[] = $sum['sum'];
    }
}
foreach($sevens as $num) {
    if($num%5 ==0) {
        $fives [] = $num;
    }
}
echo count($sumNums) . '<br>';
$count7 = count($sevens);
echo $count7 . '<br>';
$count5 = count($fives);
echo $count5. '<br>';
$prob = $count5/$count7;
echo $prob;*/

$sums = getPossibilities($moves);
$sumNums = [];
$t9 = [];
$t3 = [];

foreach ($sums as $sum) {
    $sumNums[] = $sum['sum'];
    if ($sum['sum']%29 == 0) {
        $t9[] = $sum['sum'];
    }
}
foreach($sevens as $num) {
    if($num%23 ==0) {
        $t3 [] = $num;
    }
}
echo count($sumNums) . '<br>';
$count29 = count($t9);
echo $count29 . '<br>';
$count3 = count($t3);
echo $count23. '<br>';
$prob = $count23/$count29;
echo $prob;


echo '</pre>';

exit;
?>