package tv.floe.dl4j.examples.mlp.baldridge.linear;

import static org.junit.Assert.*;

import java.io.IOException;

import org.junit.Test;

import tv.floe.dl4j.examples.mlp.baldridge.linear.BasicCSV_DataIterator;

public class TestBasicCSV_DataIterator {

	@Test
	public void test() throws IOException {
		
		BasicCSV_DataIterator iter = new BasicCSV_DataIterator( "src/test/resources/data/baldridge/linear/linear_train.txt", "", 2, 50, 1000 );
		
		iter.next();
		
		iter.next();
		
		iter.next();
		
	}

}
